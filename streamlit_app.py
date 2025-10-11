
import streamlit as st
import json, os
import matplotlib.pyplot as plt

st.set_page_config(page_title="ACE Context Demo", page_icon="🧠", layout="wide")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "playbook_history" not in st.session_state:
    st.session_state.playbook_history = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

st.title("🧠 ACE Context Demo — Continuous Chat with Playbook")

st.sidebar.header("🔑 API Key")
# API Key input
api_key_input = st.sidebar.text_input(
    "Enter your OpenAI API Key",
    type="password",
    value=st.session_state.api_key,
    help="Your API key will be used for this session only and is not stored permanently."
)

if api_key_input:
    st.session_state.api_key = api_key_input
    os.environ["OPENAI_API_KEY"] = api_key_input
    st.sidebar.success("✅ API Key set!")
elif not os.environ.get("OPENAI_API_KEY"):
    st.sidebar.warning("⚠️ Please enter your OpenAI API key to use the app.")

# Import ace_playbook after API key is potentially set
from ace_playbook import (
    retriever_topk, generator, reflector, merge_deltas,
    build_playbook_block, get_topk_by_score,
    load_all_bullets, score, bullets_by_tag, daily_counts
)

st.sidebar.markdown("---")
st.sidebar.header("Settings")
k = st.sidebar.slider("Top‑K bullets", 1, 16, 8)
retrieval_mode = st.sidebar.selectbox("Retrieval mode", ["score", "faiss"])
query_for_faiss = st.sidebar.text_input("Semantic retrieval query (optional)", "")

st.sidebar.markdown("---")
st.sidebar.write("**Playbook file**: `playbook.jsonl`")
bullets_count = len(load_all_bullets())
st.sidebar.metric("Total Bullets", bullets_count)

if st.sidebar.button("Reset Playbook"):
    if os.path.exists("playbook.jsonl"):
        os.remove("playbook.jsonl")
    st.session_state.playbook_history = []
    st.success("Playbook reset.")
    st.rerun()

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.success("Chat history cleared.")
    st.rerun()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "🎯 Context Preview", "📚 Playbook", "📊 Visualizations", "ℹ️ About"])


with tab1:
    st.subheader("💬 Continuous Chat")
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metadata" in msg and msg["metadata"]:
                with st.expander("🔍 View Details"):
                    if "topk" in msg["metadata"]:
                        st.caption("**Top-K Bullets Used:**")
                        st.code(build_playbook_block(msg["metadata"]["topk"]) or "(none)", language="markdown")
                    if "trace" in msg["metadata"]:
                        st.caption("**Trace:**")
                        st.json(msg["metadata"]["trace"])
                    if "bullets" in msg["metadata"]:
                        st.caption("**New Bullets Extracted:**")
                        st.json(msg["metadata"]["bullets"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question or give a task..."):
        # Check if API key is set
        if not st.session_state.api_key and not os.environ.get("OPENAI_API_KEY"):
            st.error("❌ Please enter your OpenAI API key in the sidebar first!")
            st.stop()
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process with ACE
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                try:
                    # Retrieve Top-K
                    topk = retriever_topk(k=k, mode=retrieval_mode, query=query_for_faiss or prompt)
                    
                    # Generate
                    g = generator(prompt, topk)
                    answer = g.get("answer", "")
                    trace = g.get("trace", [])
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Reflect and extract bullets
                    bullets = reflector(prompt, answer, trace)
                    
                    # Curate/merge into playbook
                    updated_topk = merge_deltas(bullets)
                    
                    # Store metadata
                    metadata = {
                        "topk": topk,
                        "trace": trace,
                        "bullets": bullets,
                        "updated_topk": updated_topk
                    }
                    
                    # Show expandable details
                    with st.expander("🔍 View Details"):
                        st.caption("**Top-K Bullets Used:**")
                        st.code(build_playbook_block(topk) or "(none)", language="markdown")
                        st.caption("**Trace:**")
                        st.json(trace)
                        st.caption("**New Bullets Extracted:**")
                        st.json(bullets)
                        if bullets:
                            st.success(f"✅ Added {len(bullets)} new bullet(s) to playbook!")
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": metadata
                    })
                    
                    # Track playbook growth
                    st.session_state.playbook_history.append({
                        "turn": len(st.session_state.messages) // 2,
                        "bullets_added": len(bullets),
                        "total_bullets": len(load_all_bullets())
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "metadata": {}
                    })
    
    # Show quick stats
    if st.session_state.messages:
        st.sidebar.markdown("---")
        st.sidebar.subheader("💬 Chat Stats")
        st.sidebar.metric("Total Messages", len(st.session_state.messages))
        st.sidebar.metric("User Messages", len([m for m in st.session_state.messages if m["role"] == "user"]))
        
        if st.session_state.playbook_history:
            total_bullets_added = sum(h["bullets_added"] for h in st.session_state.playbook_history)
            st.sidebar.metric("Bullets Added This Session", total_bullets_added)


with tab2:
    st.subheader("🎯 Context Preview — Next Prompt")
    
    st.markdown("""
    This shows the **Top-K bullets** that will be injected as context into the next prompt.
    These bullets guide the Generator's response based on accumulated knowledge.
    """)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        preview_k = st.slider("Preview Top-K", 1, 20, k, key="preview_k")
    with col2:
        preview_mode = st.selectbox("Mode", ["score", "faiss"], index=0 if retrieval_mode == "score" else 1, key="preview_mode")
    
    preview_query = st.text_input("Preview query (for FAISS mode)", query_for_faiss, key="preview_query")
    
    if st.button("🔄 Refresh Preview", help="Update the preview with current settings"):
        st.rerun()
    
    # Get the bullets that would be used for the next prompt
    try:
        preview_topk = retriever_topk(k=preview_k, mode=preview_mode, query=preview_query if preview_mode == "faiss" else None)
        
        if not preview_topk:
            st.info("📭 No bullets in playbook yet. Start chatting to build context!")
        else:
            # Show count and stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Bullets to Inject", len(preview_topk))
            with col2:
                avg_score = sum(score(b) for b in preview_topk) / len(preview_topk) if preview_topk else 0
                st.metric("Average Score", f"{avg_score:.2f}")
            with col3:
                total_helpful = sum(b.get("helpful", 0) for b in preview_topk)
                total_harmful = sum(b.get("harmful", 0) for b in preview_topk)
                st.metric("Total 👍/👎", f"{total_helpful}/{total_harmful}")
            
            st.markdown("---")
            
            # Show the formatted context block (as it will appear to the LLM)
            st.subheader("📋 Context Block (as sent to LLM)")
            context_block = build_playbook_block(preview_topk)
            st.code(context_block, language="markdown")
            
            st.markdown("---")
            
            # Show detailed bullet breakdown
            st.subheader("🔍 Detailed Bullet Breakdown")
            
            for i, bullet in enumerate(preview_topk, 1):
                bullet_score = score(bullet)
                helpful = bullet.get("helpful", 0)
                harmful = bullet.get("harmful", 0)
                tags = bullet.get("tags", [])
                content = bullet.get("content", "")
                last_seen = bullet.get("last_seen", "N/A")
                
                # Color code by score
                if bullet_score > 5:
                    emoji = "🟢"
                elif bullet_score > 0:
                    emoji = "🟡"
                elif bullet_score == 0:
                    emoji = "⚪"
                else:
                    emoji = "🔴"
                
                with st.expander(f"{emoji} **Bullet #{i}** — Score: {bullet_score} | 👍 {helpful} | 👎 {harmful}"):
                    st.markdown(f"**Content:** {content}")
                    if tags:
                        st.markdown(f"**Tags:** {', '.join(tags)}")
                    st.caption(f"Last seen: {last_seen}")
                    
                    # Show relevance indicator
                    if bullet_score > 3:
                        st.success("✨ High-value bullet - frequently helpful")
                    elif bullet_score < 0:
                        st.warning("⚠️ Potentially harmful - use with caution")
    
    except Exception as e:
        st.error(f"Error loading preview: {str(e)}")


with tab3:
    st.subheader("📚 Playbook View")
    
    bullets = load_all_bullets()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption(f"**{len(bullets)} bullets** in playbook (score-sorted)")
    with col2:
        view_mode = st.selectbox("View", ["Top 20", "All", "Top 10", "Top 50"], index=0)
    
    if not bullets:
        st.info("📭 No bullets yet — start chatting to build your playbook!")
    else:
        # Determine how many to show
        limit_map = {"Top 10": 10, "Top 20": 20, "Top 50": 50, "All": 1000}
        limit = limit_map.get(view_mode, 20)
        
        display_bullets = get_topk_by_score(limit)
        
        # Display as expandable cards
        for i, bullet in enumerate(display_bullets, 1):
            bullet_score = score(bullet)
            helpful = bullet.get("helpful", 0)
            harmful = bullet.get("harmful", 0)
            tags = bullet.get("tags", [])
            content = bullet.get("content", "")
            last_seen = bullet.get("last_seen", "N/A")
            
            # Color code by score
            if bullet_score > 5:
                emoji = "🟢"
            elif bullet_score > 0:
                emoji = "🟡"
            elif bullet_score == 0:
                emoji = "⚪"
            else:
                emoji = "🔴"
            
            with st.expander(f"{emoji} **#{i}** — Score: {bullet_score} | 👍 {helpful} | 👎 {harmful}"):
                st.markdown(f"**Content:** {content}")
                if tags:
                    st.markdown(f"**Tags:** {', '.join(tags)}")
                st.caption(f"Last seen: {last_seen}")
        
        # Show playbook growth over session
        if st.session_state.playbook_history:
            st.markdown("---")
            st.subheader("📈 Playbook Growth This Session")
            
            import matplotlib.pyplot as plt
            
            turns = [h["turn"] for h in st.session_state.playbook_history]
            totals = [h["total_bullets"] for h in st.session_state.playbook_history]
            
            fig = plt.figure(figsize=(8, 3))
            plt.plot(turns, totals, marker="o", linewidth=2, markersize=6)
            plt.xlabel("Turn Number")
            plt.ylabel("Total Bullets")
            plt.title("Playbook Growth")
            plt.grid(True, alpha=0.3)
            st.pyplot(fig, clear_figure=True)

with tab4:
    st.subheader("📊 Visualizations")

    bullets = load_all_bullets()
    if not bullets:
        st.info("No bullets yet — run a turn first.")
    else:
        # 1) Top 10 bullets by score (bar chart)
        st.markdown("**Top 10 bullets by score (helpful − harmful)**")
        sorted_b = sorted(bullets, key=score, reverse=True)[:10]
        labels = [b.get("content","")[:40] + ("…" if len(b.get('content',''))>40 else "") for b in sorted_b]
        values = [score(b) for b in sorted_b]

        fig1 = plt.figure()
        plt.barh(labels, values)
        plt.xlabel("Score")
        plt.ylabel("Bullet")
        plt.gca().invert_yaxis()
        st.pyplot(fig1, clear_figure=True)

        # 2) Helpful vs Harmful totals
        st.markdown("**Helpful vs Harmful totals**")
        total_helpful = sum(b.get("helpful",0) for b in bullets)
        total_harmful = sum(b.get("harmful",0) for b in bullets)

        fig2 = plt.figure()
        plt.bar(["Helpful","Harmful"], [total_helpful, total_harmful])
        plt.ylabel("Count")
        st.pyplot(fig2, clear_figure=True)

        # 3) Tags frequency
        st.markdown("**Tag frequency**")
        tag_counts = bullets_by_tag()
        if tag_counts:
            tags = list(tag_counts.keys())
            counts = list(tag_counts.values())
            fig3 = plt.figure()
            plt.bar(tags, counts)
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig3, clear_figure=True)
        else:
            st.caption("No tags found yet.")

        # 4) Daily additions
        st.markdown("**Bullets added/updated by day**")
        by_day = daily_counts()
        if by_day:
            days = sorted(by_day.keys())
            counts = [by_day[d] for d in days]
            fig4 = plt.figure()
            plt.plot(days, counts, marker="o")
            plt.xlabel("Day")
            plt.ylabel("Events")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig4, clear_figure=True)
        else:
            st.caption("No last_seen timestamps yet.")

with tab5:
    st.markdown("""
## 🧠 ACE Context Demo — Continuous Chat

**How it works**
1. **Chat Interface**: Ask questions or give tasks in natural language
2. **Retriever** gets Top‑K bullets from the JSON Playbook (score or FAISS)
3. **Generator** solves the task with Top‑K tips injected
4. **Reflector** turns the run into new bullets (strategies, pitfalls, guardrails)
5. **Curator** merges & dedups bullets, bumps helpful/harmful, updates the Playbook
6. **Continuous Learning**: Each chat turn enriches the playbook

**Features**
- 💬 **Continuous Chat**: Chat history persists during your session
- 🎯 **Context Preview**: See exactly which bullets will be sent with the next prompt
- 📚 **Live Playbook**: See bullets grow with each interaction
- 📊 **Visualizations**: Track playbook statistics and growth
- 🔍 **Expandable Details**: View Top-K bullets, traces, and new bullets for each turn
- ⚙️ **Configurable**: Adjust Top-K size, retrieval mode (score/FAISS)

**Tabs**
- **💬 Chat**: Interactive conversation with the ACE agent
- **🎯 Context Preview**: Preview the Top-K bullets that will be injected into the next prompt
- **📚 Playbook**: Browse all bullets, sorted by score
- **📊 Visualizations**: Charts showing playbook statistics and growth
- **ℹ️ About**: This information page

**Visualizations**
- Top bullets by score
- Helpful vs Harmful totals
- Tags frequency
- Daily timeline of playbook updates
- Playbook growth over session

**Tips**
- Start chatting to build your playbook from scratch
- Use the **Context Preview** tab to see what knowledge will guide the next response
- Use the sidebar to adjust Top-K and retrieval settings
- Click "View Details" on any message to see the ACE pipeline in action
- Reset the playbook to start fresh, or clear chat history to begin a new session

**Environment**
- Model: `gpt-4o-mini`
- API Key: Enter via sidebar, or loaded from Streamlit secrets/environment variable

> Built with Streamlit, LangChain, OpenAI, and matplotlib
""")
