
# ACE Context (Retriever → Reflector → Curator) — Streamlit Demo

This demo shows a simple **ACE** loop with a JSONL Playbook, a **Streamlit** UI,
and optional **FAISS**-based semantic retrieval for Top‑K bullets.

## Features
- View Playbook bullets (JSONL)
- Run a "Generator" turn (uses OpenAI chat model) with **Top‑K** bullets injected
- Reflect on the run to extract new bullets
- Curate/merge bullets with dedup and helpful/harmful counters
- **Optional:** enable FAISS semantic retrieval for Top‑K selection

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# set your key
export OPENAI_API_KEY=sk-...      # (Windows PowerShell: $env:OPENAI_API_KEY="sk-...")

streamlit run streamlit_app.py
```

Open the URL shown by Streamlit (usually http://localhost:8501).

## Files
- `ace_playbook.py` — Playbook store, FAISS retriever, generator/reflector/curator functions
- `streamlit_app.py` — UI to run a turn and manage bullets
- `requirements.txt` — dependencies

## Notes
- The demo writes/reads `playbook.jsonl` in the current directory.
- FAISS is optional; toggle it in the UI. If disabled, Top‑K is rank‑sorted by (helpful−harmful).
- This is a teaching/reference implementation — adjust for production (PII scrubbing, auth, queues, etc).
