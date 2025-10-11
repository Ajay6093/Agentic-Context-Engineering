
# Agentic Context Engineering Context Demo — Continuous Chat with Playbook

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

**Required: Enter your API key in the app's sidebar**

When you run the app, you'll see a password input field in the sidebar. Simply enter your OpenAI API key there. The key is:
- ✅ Only stored for your current session
- ✅ Not saved to disk or committed to git
- ✅ Cleared when you close the browser/tab

**Alternative: Environment Variable (Advanced)**

For development, you can set an environment variable before running:
```bash
export OPENAI_API_KEY=sk-your-key-here  # Linux/Mac
# OR
$env:OPENAI_API_KEY="sk-your-key-here"  # Windows PowerShell
```

**For Streamlit Cloud Deployment:**

Add your API key in the Streamlit Cloud dashboard under "Settings" → "Secrets":
```toml
OPENAI_API_KEY = "sk-your-actual-api-key"
```

⚠️ **Important:** Never commit your API key to git!

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
