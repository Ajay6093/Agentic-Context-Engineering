
"""
ACE Playbook - Core Implementation

This module implements the ACE (Adaptive Context Engine) framework as described in the research paper.
It provides functions for storing, retrieving, and managing contextual "bullets" (learnings) that
improve over time through conversation.

Key Components:
1. Storage: JSONL-based persistence of playbook bullets
2. Retrieval: Score-based and FAISS semantic retrieval of relevant bullets
3. Generator: LLM that uses retrieved bullets to answer queries
4. Reflector: LLM that extracts learnings from conversations into new bullets
5. Curator: Merges and deduplicates bullets, updating helpful/harmful scores

The ACE loop: Retrieve → Generate → Reflect → Curate → Store
"""

import os, json, time, uuid, math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ============================================================================
# API Key Configuration
# ============================================================================
# Load API key from Streamlit secrets or environment variable
# This allows the app to work in multiple deployment scenarios:
# 1. Streamlit Cloud with secrets configured
# 2. Local development with environment variables
# 3. Frontend password input (sets environment variable at runtime)
try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass  # Fall back to environment variable


# ============================================================================
# Storage Layer - JSONL Persistence
# ============================================================================
PLAYBOOK_PATH = "playbook.jsonl"

def _load() -> List[Dict]:
    """
    Load all bullets from the playbook JSONL file.
    
    Returns:
        List of bullet dictionaries. Each bullet contains:
        - id: unique identifier (UUID)
        - content: the actual learning/strategy/pitfall text
        - tags: list of category tags
        - helpful: count of times this bullet was marked helpful
        - harmful: count of times this bullet was marked harmful
        - last_seen: ISO timestamp of last update
    
    Implementation Notes:
        - Uses JSONL (JSON Lines) format: one JSON object per line
        - Each line is a separate bullet, making append operations efficient
        - Returns empty list if file doesn't exist (cold start)
    """
    if not os.path.exists(PLAYBOOK_PATH): 
        return []
    with open(PLAYBOOK_PATH, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _save(bullets: List[Dict]):
    """
    Save all bullets to the playbook JSONL file.
    
    Args:
        bullets: List of bullet dictionaries to persist
    
    Implementation Notes:
        - Overwrites the entire file (not append-only)
        - This allows for deduplication and reordering
        - Uses ensure_ascii=False to support Unicode characters
        - Each bullet is written as a single line of JSON
    """
    with open(PLAYBOOK_PATH, "w", encoding="utf-8") as f:
        for b in bullets:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")


# ============================================================================
# Retrieval Layer - Score-based and Semantic Search
# ============================================================================

def get_topk_by_score(k: int = 8) -> List[Dict]:
    """
    Retrieve top-K bullets ranked by their score (helpful - harmful).
    
    Args:
        k: Number of top bullets to return
    
    Returns:
        List of top K bullets sorted by score (descending)
    
    ACE Framework Context:
        This is the SCORE-BASED RETRIEVAL strategy mentioned in the paper.
        Bullets with higher (helpful - harmful) scores are prioritized.
        This implements a simple but effective relevance ranking based on
        community feedback (upvotes/downvotes analogy).
    
    Example:
        Bullet with helpful=10, harmful=2 has score=8
        Bullet with helpful=5, harmful=1 has score=4
        First bullet would be ranked higher
    """
    bullets = _load()
    # Sort by score: (helpful - harmful), highest first
    bullets.sort(key=lambda b: (b.get("helpful",0) - b.get("harmful",0)), reverse=True)
    return bullets[:k]


def merge_deltas(deltas: List[Dict]) -> List[Dict]:
    """
    Merge new bullets (deltas) into the existing playbook (CURATOR role).
    
    Args:
        deltas: List of new bullets extracted by the Reflector
    
    Returns:
        Updated top-K bullets after merging
    
    ACE Framework Context:
        This is the CURATOR function from the paper. It performs:
        1. Deduplication: If bullet content already exists, increment its vote count
        2. Addition: If bullet is new, add it with initial vote counts
        3. Timestamp Update: Track when bullets were last seen/used
        
    Workflow:
        - Load existing playbook
        - For each new bullet (delta):
            * Check if identical content already exists
            * If exists: increment helpful/harmful counter, update timestamp
            * If new: create new bullet with UUID, initialize counters
        - Save updated playbook
        - Return top-K by score
    
    This implements the "merge and deduplicate" strategy that prevents
    playbook bloat while reinforcing frequently useful patterns.
    """
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    bullets = _load()
    
    for d in deltas:
        vote = d.get("vote", "helpful")  # Default to helpful if not specified
        content = d.get("content", "").strip()
        
        if not content:  # Skip empty bullets
            continue
        
        # Check for duplicate: exact content match
        found = next((b for b in bullets if b.get("content", "").strip() == content), None)
        
        if found:
            # Bullet exists: increment the appropriate counter
            found[vote] = found.get(vote, 0) + 1
            found["last_seen"] = now
        else:
            # New bullet: create and add to playbook
            bullets.append({
                "id": str(uuid.uuid4()),
                "content": content,
                "tags": d.get("tags", []),
                "helpful": 1 if vote == "helpful" else 0,
                "harmful": 1 if vote == "harmful" else 0,
                "last_seen": now
            })
    
    _save(bullets)
    return get_topk_by_score()


# ============================================================================
# FAISS Semantic Retrieval (Optional Advanced Feature)
# ============================================================================
# Try to import FAISS for semantic search capability
# FAISS = Facebook AI Similarity Search, enables vector-based retrieval
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


def _faiss_topk(k: int = 8, query: Optional[str] = None) -> List[Dict]:
    """
    Retrieve top-K bullets using semantic similarity (FAISS vector search).
    
    Args:
        k: Number of bullets to retrieve
        query: User's query text for semantic matching
    
    Returns:
        List of K bullets most semantically similar to the query
    
    ACE Framework Context:
        This implements SEMANTIC RETRIEVAL as an alternative to score-based.
        Uses OpenAI embeddings to convert bullets into vectors, then finds
        the most similar bullets to the query using cosine similarity.
    
    Workflow:
        1. Embed all bullet contents into vectors using OpenAI embeddings
        2. Build FAISS index with normalized vectors (for cosine similarity)
        3. Embed the user query
        4. Search index for K nearest neighbors
        5. Return the corresponding bullets
    
    Advantages over score-based:
        - Context-aware: finds bullets relevant to current query topic
        - Semantic understanding: matches meaning, not just keywords
    
    Disadvantages:
        - Requires API calls for embeddings (cost, latency)
        - Builds index on every call (not efficient for large playbooks)
    
    Note: For production, you'd want to persist the FAISS index and
    update it incrementally rather than rebuilding each time.
    """
    if not HAS_FAISS:
        # Fallback to score-based if FAISS not available
        return get_topk_by_score(k)
    
    bullets = _load()
    if not bullets:
        return []
    
    # Extract text content from all bullets
    texts = [b.get("content", "") for b in bullets]
    
    # Create embeddings using OpenAI (uses OPENAI_API_KEY from environment)
    embed = OpenAIEmbeddings()
    vecs = embed.embed_documents(texts)
    
    # Build FAISS index for inner product search
    dim = len(vecs[0])  # Dimension of embedding vectors
    index = faiss.IndexFlatIP(dim)  # IP = Inner Product
    
    import numpy as np
    mat = np.array(vecs, dtype="float32")
    
    # Normalize vectors so inner product = cosine similarity
    faiss.normalize_L2(mat)
    index.add(mat)
    
    if not query or not query.strip():
        # No query provided: fallback to score-based ranking
        return get_topk_by_score(k)
    
    # Embed the query
    q = embed.embed_query(query)
    q = np.array([q], dtype="float32")
    faiss.normalize_L2(q)
    
    # Search for K nearest neighbors
    D, I = index.search(q, min(k, len(bullets)))
    
    # Return bullets at the found indices
    out = [bullets[int(i)] for i in I[0] if int(i) >= 0]
    return out


# ============================================================================
# LLM Components - Generator and Reflector
# ============================================================================
# Lazy initialization of LLM instances to support runtime API key configuration
# This prevents errors when the module is imported before API key is set

_llm_gen = None  # Generator LLM instance (cached)
_llm_ref = None  # Reflector LLM instance (cached)


def _get_llm_gen():
    """
    Lazy initialization of Generator LLM.
    
    Returns:
        ChatOpenAI instance configured for generation tasks
    
    Implementation Notes:
        - Only creates the LLM instance on first use
        - Allows API key to be set at runtime (e.g., via frontend input)
        - Uses gpt-4o-mini for cost efficiency
        - Temperature=0 for consistent, deterministic responses
    """
    global _llm_gen
    if _llm_gen is None:
        _llm_gen = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _llm_gen


def _get_llm_ref():
    """
    Lazy initialization of Reflector LLM.
    
    Returns:
        ChatOpenAI instance configured for reflection tasks
    
    Implementation Notes:
        - Separate instance from Generator (could use different models/params)
        - Currently uses same model (gpt-4o-mini) but architecture supports
          using different models for different roles
    """
    global _llm_ref
    if _llm_ref is None:
        _llm_ref = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _llm_ref


def build_playbook_block(topk: List[Dict]) -> str:
    """
    Format top-K bullets into a markdown block for injection into prompts.
    
    Args:
        topk: List of bullet dictionaries to format
    
    Returns:
        Formatted markdown string with bullets as a bulleted list
    
    ACE Framework Context:
        This is how retrieved context is INJECTED into the Generator's prompt.
        The playbook bullets serve as "tips" or "guardrails" that guide the
        LLM's response based on accumulated learnings.
    
    Example Output:
        ### ACE Playbook (Top-K)
        - When planning events, always consider budget constraints first
        - Use specific examples to illustrate abstract concepts
        - Break down complex tasks into smaller steps
    """
    if not topk: 
        return ""
    lines = "\n".join(f"- {b.get('content','')}" for b in topk)
    return f"### ACE Playbook (Top-K)\n{lines}\n"


def generator(user_query: str, topk: List[Dict], conversation_history: Optional[List[Dict]] = None) -> Dict:
    """
    GENERATOR: Answer user queries using playbook context and conversation history.
    
    Args:
        user_query: The current user question/task
        topk: Top-K retrieved bullets to use as context
        conversation_history: Previous messages in the conversation
    
    Returns:
        Dictionary with:
        - answer: The LLM's response (string)
        - trace: List of reasoning steps taken
    
    ACE Framework Context:
        This is the GENERATOR component from the paper. It:
        1. Receives retrieved playbook bullets (Top-K)
        2. Incorporates conversation history for continuity
        3. Uses the playbook as "guidelines" to inform its response
        4. Returns both the answer and a trace of its reasoning
    
    Workflow:
        1. Build system message with playbook bullets injected
        2. Add conversation history to maintain context
        3. Add current user query
        4. Invoke LLM to generate response
        5. Parse JSON response (with error handling)
        6. Return structured output
    
    JSON Response Format:
        {
          "answer": "The helpful response as a string",
          "trace": ["step 1", "step 2", "step 3"]
        }
    
    Implementation Notes:
        - Explicitly requires JSON output to enable structured parsing
        - Maintains full conversation context (not just last message)
        - Handles edge cases where LLM returns invalid JSON
        - Converts non-string answers to strings (e.g., bare numbers)
    """
    # System message with instructions and playbook context
    system_msg = (
        "You are the GENERATOR - an AI assistant that helps users with their tasks.\n"
        "Use the ACE Playbook if relevant.\n"
        "Maintain conversation context and refer to previous messages when appropriate.\n\n"
        "IMPORTANT: You MUST respond ONLY with valid JSON in this exact format:\n"
        "{\n"
        '  "answer": "your helpful response here as a string",\n'
        '  "trace": ["step 1", "step 2", "step 3"]\n'
        "}\n\n"
        "Do not include any text before or after the JSON.\n"
        "The answer field must be a string, even for numerical results.\n"
        "Example for math: {\"answer\": \"The result is 42\", \"trace\": [\"Added 15 + 27\"]}"
    )
    ctx = build_playbook_block(topk)
    
    llm_gen = _get_llm_gen()  # Get LLM instance (lazy initialization)
    
    # Build complete message history for the LLM
    messages = [("system", system_msg + "\n\n" + ctx)]
    
    # Add previous conversation turns
    if conversation_history:
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(("user", content))
            elif role == "assistant":
                # For assistant messages, only include the text content
                # (not the original JSON structure with trace)
                messages.append(("assistant", content))
    
    # Add the current user query
    messages.append(("user", user_query))
    
    # Invoke the LLM
    res = llm_gen.invoke(messages).content
    
    # Clean up response - remove markdown code blocks if LLM wrapped the JSON
    res = res.strip()
    if res.startswith("```json"):
        res = res[7:]
    if res.startswith("```"):
        res = res[3:]
    if res.endswith("```"):
        res = res[:-3]
    res = res.strip()
    
    # Parse JSON with error handling
    try:
        parsed = json.loads(res)
        
        # Ensure answer is always a string (convert numbers if needed)
        if "answer" in parsed and not isinstance(parsed["answer"], str):
            parsed["answer"] = str(parsed["answer"])
        
        # Ensure trace exists (default to empty array)
        if "trace" not in parsed:
            parsed["trace"] = []
        
        return parsed
    except json.JSONDecodeError as e:
        # Fallback: if JSON parsing fails, wrap the raw response
        # This prevents crashes and allows the conversation to continue
        return {
            "answer": res,
            "trace": [f"Raw response (JSON parse failed): {str(e)}"]
        }


def reflector(user_query: str, answer: str, trace: List[str]) -> List[Dict]:
    """
    REFLECTOR: Extract learnings from a conversation turn into playbook bullets.
    
    Args:
        user_query: The user's original question/task
        answer: The generated answer from the Generator
        trace: The reasoning steps from the Generator
    
    Returns:
        List of new bullet dictionaries, each containing:
        - content: The learning/strategy/pitfall text
        - tags: List of category tags
        - vote: "helpful" or "harmful"
    
    ACE Framework Context:
        This is the REFLECTOR component from the paper. It:
        1. Analyzes a completed conversation turn
        2. Extracts reusable insights, strategies, and pitfalls
        3. Returns 2-6 concise bullets that can be added to the playbook
        4. Categorizes each bullet as "helpful" or "harmful"
    
    Purpose:
        The Reflector enables the system to LEARN from interactions.
        Over time, the playbook grows to include:
        - Successful strategies ("helpful")
        - Common pitfalls to avoid ("harmful")
        - Domain-specific guidelines
        - User preferences and patterns
    
    Workflow:
        1. Package the query, answer, and trace into a JSON payload
        2. Prompt the Reflector LLM to extract learnings
        3. Parse the JSON response
        4. Return bullets for the Curator to merge
    
    JSON Response Format:
        {
          "bullets": [
            {
              "content": "Strategy or learning text",
              "tags": ["category1", "category2"],
              "vote": "helpful"
            },
            ...
          ]
        }
    
    Implementation Notes:
        - Returns empty list if extraction fails (graceful degradation)
        - Handles malformed JSON responses
        - Tags enable categorical organization of bullets
        - Vote field enables ranking by community feedback analogy
    """
    system_msg = (
        "You are the REFLECTOR.\n"
        "Extract 2–6 concise, reusable bullets (strategy/pitfall/guardrail).\n\n"
        "IMPORTANT: You MUST respond ONLY with valid JSON in this exact format:\n"
        "{\n"
        '  "bullets": [\n'
        '    {"content": "bullet text here", "tags": ["tag1", "tag2"], "vote": "helpful"},\n'
        '    {"content": "another bullet", "tags": ["tag3"], "vote": "harmful"}\n'
        "  ]\n"
        "}\n\n"
        'vote must be either "helpful" or "harmful".\n'
        "Do not include any text before or after the JSON."
    )
    
    # Package the conversation turn for analysis
    payload = json.dumps({
        "query": user_query,
        "answer": answer,
        "trace": trace
    }, indent=2)
    
    llm_ref = _get_llm_ref()  # Get Reflector LLM instance
    res = llm_ref.invoke([("system", system_msg), ("user", payload)]).content
    
    # Clean up response - remove markdown code blocks if present
    res = res.strip()
    if res.startswith("```json"):
        res = res[7:]
    if res.startswith("```"):
        res = res[3:]
    if res.endswith("```"):
        res = res[:-3]
    res = res.strip()
    
    # Parse JSON with error handling
    try:
        parsed = json.loads(res)
        return parsed.get("bullets", [])
    except json.JSONDecodeError as e:
        # Fallback: return empty bullets if parsing fails
        # This prevents the reflection step from breaking the entire flow
        print(f"Reflector JSON parse error: {e}")
        return []


def retriever_topk(k: int = 8, mode: str = "score", query: Optional[str] = None) -> List[Dict]:
    """
    RETRIEVER: Fetch top-K bullets from playbook using specified strategy.
    
    Args:
        k: Number of bullets to retrieve
        mode: Retrieval strategy - "score" or "faiss"
        query: User query (used for semantic search if mode="faiss")
    
    Returns:
        List of K bullets most relevant according to the chosen strategy
    
    ACE Framework Context:
        This is the RETRIEVER component from the paper. It selects which
        bullets from the playbook should be injected into the Generator's
        context for the current query.
    
    Retrieval Strategies:
        1. "score": Rank by (helpful - harmful) score
           - Simple, fast, no API calls
           - Prioritizes consistently useful bullets
           - Topic-agnostic
        
        2. "faiss": Semantic similarity search
           - Context-aware, finds topically relevant bullets
           - Uses vector embeddings and cosine similarity
           - Requires OpenAI API calls (cost/latency trade-off)
    
    Implementation Notes:
        - Delegates to _faiss_topk() or get_topk_by_score()
        - Can be extended with hybrid strategies
        - Future: BM25, keyword matching, temporal decay, etc.
    """
    if mode == "faiss":
        return _faiss_topk(k=k, query=query)
    return get_topk_by_score(k=k)


# ============================================================================
# Utility Functions for Visualization and Analytics
# ============================================================================

def load_all_bullets():
    """
    Load all bullets from playbook without filtering or sorting.
    
    Returns:
        Complete list of all bullets in the playbook
    
    Use Cases:
        - Visualization dashboards
        - Analytics and statistics
        - Bulk export
        - Debugging and inspection
    """
    return _load()


def score(b):
    """
    Calculate the score of a bullet (helpful - harmful).
    
    Args:
        b: Bullet dictionary
    
    Returns:
        Integer score (can be negative if harmful > helpful)
    
    ACE Framework Context:
        This simple scoring function implements a voting mechanism.
        Bullets that consistently help get higher scores.
        Bullets that lead to problems get negative scores.
        
    Example:
        helpful=10, harmful=2 → score=8 (good bullet)
        helpful=3, harmful=8 → score=-5 (harmful bullet)
    """
    return b.get("helpful", 0) - b.get("harmful", 0)


def bullets_by_tag():
    """
    Count bullets by their tags for categorical analysis.
    
    Returns:
        Dictionary mapping tag names to counts
        Example: {"planning": 5, "math": 3, "debugging": 2}
    
    Use Cases:
        - Tag frequency visualization
        - Identify dominant categories
        - Balance checking (ensure diverse coverage)
        - Topic-based filtering UI
    
    Implementation:
        - Flattens all tags from all bullets
        - Uses Counter to aggregate
        - Filters out empty tags
    """
    from collections import Counter
    tags = []
    for b in _load():
        tags.extend(b.get("tags", []) or [])
    c = Counter([t for t in tags if t])
    return dict(c)


def daily_counts():
    """
    Count bullet updates by date for temporal analysis.
    
    Returns:
        Dictionary mapping dates (YYYY-MM-DD) to event counts
        Example: {"2025-10-10": 5, "2025-10-11": 3}
    
    Use Cases:
        - Playbook growth visualization
        - Activity timeline
        - Identify learning spikes
        - Session analysis
    
    Implementation:
        - Extracts date from last_seen ISO timestamp
        - Groups by date and counts
        - Returns dict for easy plotting
    
    Note: "last_seen" is updated both when a bullet is created
    and when it's reinforced (duplicate content matched)
    """
    from collections import Counter
    dates = []
    for b in _load():
        ls = b.get("last_seen")
        if not ls:
            continue
        try:
            # Extract date part from ISO timestamp (YYYY-MM-DDTHH:MM:SSZ)
            d = ls.split("T")[0]
            dates.append(d)
        except Exception:
            pass  # Skip malformed timestamps
    return dict(Counter(dates))

