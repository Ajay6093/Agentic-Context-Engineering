
# ACE Context Demo — Continuous Chat with Playbook

This demo shows a **continuous chat interface** with an **ACE** (Adaptive Context Engine) loop that builds a JSONL Playbook dynamically as you interact with it. Features include:
- 💬 **Continuous chat** with conversation history
- 📚 **Live playbook** that grows with each interaction
- 📊 **Real-time visualizations** of playbook statistics
- 🔍 **Detailed trace views** for each ACE pipeline run

## Features
- **Chat Interface**: Natural conversation with the AI assistant
- **Dynamic Playbook**: Automatically extracts and stores reusable bullets (strategies, pitfalls, guardrails)
- **Retrieval Modes**: Score-based or FAISS semantic retrieval for Top-K bullets
- **Reflection**: Each turn generates new insights that enrich the playbook
- **Curation**: Automatic deduplication and ranking of bullets by helpful/harmful votes
- **Visualizations**: Track playbook growth, tag frequencies, and helpful/harmful ratios


## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### Setting up your OpenAI API Key

**Option 1: Environment Variable (Local Development)**
```bash
export OPENAI_API_KEY=sk-...      # (Windows PowerShell: $env:OPENAI_API_KEY="sk-...")
```

**Option 2: Streamlit Secrets (Recommended for Deployment)**

Create `.streamlit/secrets.toml` and add:
```toml
OPENAI_API_KEY = "sk-your-api-key-here"
```

⚠️ **Important:** Never commit your API key to git! The `.gitignore` file already excludes `secrets.toml`.

For **Streamlit Cloud**, add your secrets in the app dashboard under "Settings" → "Secrets".

### Run the app

```bash
streamlit run streamlit_app.py
```

Open the URL shown by Streamlit (usually http://localhost:8501).

## Files
- `ace_playbook.py` — Playbook store, FAISS retriever, generator/reflector/curator functions
- `streamlit_app.py` — Continuous chat UI with live playbook visualization
- `requirements.txt` — dependencies
- `.streamlit/secrets.toml` — (create this) for your OpenAI API key

## How It Works

### The ACE Loop
1. **User Input**: You chat with the assistant
2. **Retriever**: Gets Top-K most relevant bullets from the playbook
3. **Generator**: Uses OpenAI to answer your query, informed by the Top-K bullets
4. **Reflector**: Analyzes the interaction and extracts 2-6 reusable bullets
5. **Curator**: Merges new bullets into the playbook, deduplicates, and ranks by score

### Continuous Learning
Each chat turn enriches the playbook. Over time, the system builds up a knowledge base of strategies, pitfalls, and guardrails specific to your domain and usage patterns.


## Notes
- The demo writes/reads `playbook.jsonl` in the current directory.
- FAISS is optional; toggle it in the UI. If disabled, Top‑K is rank‑sorted by (helpful−harmful).
- This is a teaching/reference implementation — adjust for production (PII scrubbing, auth, queues, etc).
