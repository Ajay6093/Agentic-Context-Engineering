
import os, json, time, uuid, math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load API key from Streamlit secrets or environment
try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass  # Fall back to environment variable

# -------- Storage (JSONL) --------
PLAYBOOK_PATH = "playbook.jsonl"

def _load() -> List[Dict]:
    if not os.path.exists(PLAYBOOK_PATH): return []
    with open(PLAYBOOK_PATH, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def _save(bullets: List[Dict]):
    with open(PLAYBOOK_PATH, "w", encoding="utf-8") as f:
        for b in bullets:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")

def get_topk_by_score(k: int = 8) -> List[Dict]:
    bullets = _load()
    bullets.sort(key=lambda b: (b.get("helpful",0)-b.get("harmful",0)), reverse=True)
    return bullets[:k]

def merge_deltas(deltas: List[Dict]) -> List[Dict]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    bullets = _load()
    for d in deltas:
        vote = d.get("vote","helpful")
        content = d.get("content","").strip()
        if not content: 
            continue
        found = next((b for b in bullets if b.get("content","").strip()==content), None)
        if found:
            found[vote] = found.get(vote,0) + 1
            found["last_seen"] = now
        else:
            bullets.append({
                "id": str(uuid.uuid4()),
                "content": content,
                "tags": d.get("tags", []),
                "helpful": 1 if vote=="helpful" else 0,
                "harmful": 1 if vote=="harmful" else 0,
                "last_seen": now
            })
    _save(bullets)
    return get_topk_by_score()

# -------- FAISS semantic retrieval (optional) --------
# Build vectors on the fly (small demo). For larger stores, persist the index.
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

def _faiss_topk(k: int = 8, query: Optional[str] = None) -> List[Dict]:
    if not HAS_FAISS:
        return get_topk_by_score(k)
    bullets = _load()
    if not bullets:
        return []
    texts = [b.get("content","") for b in bullets]
    embed = OpenAIEmbeddings()  # uses OPENAI_API_KEY
    vecs = embed.embed_documents(texts)
    dim = len(vecs[0])
    index = faiss.IndexFlatIP(dim)
    import numpy as np
    mat = np.array(vecs, dtype="float32")
    # Normalize for inner product == cosine
    faiss.normalize_L2(mat)
    index.add(mat)

    if not query or not query.strip():
        # fallback: return by score
        return get_topk_by_score(k)

    q = embed.embed_query(query)
    q = np.array([q], dtype="float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, min(k, len(bullets)))
    # order by distance desc
    out = [bullets[int(i)] for i in I[0] if int(i) >= 0]
    return out

# -------- LLM roles (Generator & Reflector) --------
# Lazy initialization to allow API key to be set at runtime
_llm_gen = None
_llm_ref = None

def _get_llm_gen():
    """Lazy initialization of Generator LLM"""
    global _llm_gen
    if _llm_gen is None:
        _llm_gen = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _llm_gen

def _get_llm_ref():
    """Lazy initialization of Reflector LLM"""
    global _llm_ref
    if _llm_ref is None:
        _llm_ref = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _llm_ref

def build_playbook_block(topk: List[Dict]) -> str:
    if not topk: return ""
    lines = "\n".join(f"- {b.get('content','')}" for b in topk)
    return f"### ACE Playbook (Top-K)\n{lines}\n"

def generator(user_query: str, topk: List[Dict]) -> Dict:
    system_msg = (
        "You are the GENERATOR.\n"
        "Use the ACE Playbook if relevant.\n"
        "Return JSON with keys: answer (string), trace (array of steps)."
    )
    ctx = build_playbook_block(topk)
    llm_gen = _get_llm_gen()  # Get LLM instance only when needed
    res = llm_gen.invoke([
        ("system", system_msg + "\n\n" + ctx),
        ("user", f"Task: {user_query}\nReturn JSON only.")
    ]).content
    return json.loads(res)

def reflector(user_query: str, answer: str, trace: List[str]) -> List[Dict]:
    system_msg = (
        "You are the REFLECTOR.\n"
        "Extract 2–6 concise, reusable bullets (strategy/pitfall/guardrail).\n"
        "Return JSON: {\"bullets\":[{\"content\":\"...\",\"tags\":[\"...\"],\"vote\":\"helpful|harmful\"}]}"
    )
    payload = json.dumps({"query": user_query, "answer": answer, "trace": trace}, indent=2)
    llm_ref = _get_llm_ref()  # Get LLM instance only when needed
    res = llm_ref.invoke([("system", system_msg), ("user", payload)]).content
    return json.loads(res).get("bullets", [])

def retriever_topk(k: int = 8, mode: str = "score", query: Optional[str] = None) -> List[Dict]:
    if mode == "faiss":
        return _faiss_topk(k=k, query=query)
    return get_topk_by_score(k=k)



# -------- Stats helpers (for visualizations) --------
def load_all_bullets():
    return _load()

def score(b):
    return b.get("helpful",0) - b.get("harmful",0)

def bullets_by_tag():
    from collections import Counter
    tags = []
    for b in _load():
        tags.extend(b.get("tags",[]) or [])
    from collections import Counter
    c = Counter([t for t in tags if t])
    return dict(c)

def daily_counts():
    # returns dict: date -> count
    from collections import Counter
    import datetime as _dt
    dates = []
    for b in _load():
        ls = b.get("last_seen")
        if not ls: 
            continue
        try:
            d = ls.split("T")[0]
            dates.append(d)
        except Exception:
            pass
    return dict(Counter(dates))
