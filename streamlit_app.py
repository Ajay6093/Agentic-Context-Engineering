
import streamlit as st
import json, os
import matplotlib.pyplot as plt

from ace_playbook import (
    retriever_topk, generator, reflector, merge_deltas,
    build_playbook_block, get_topk_by_score,
    load_all_bullets, score, bullets_by_tag, daily_counts
)

st.set_page_config(page_title="ACE Context Demo", page_icon="🧠", layout="wide")

st.title("🧠 ACE Context Demo — Retriever → Reflector → Curator")

st.sidebar.header("Settings")
k = st.sidebar.slider("Top‑K bullets", 1, 16, 8)
retrieval_mode = st.sidebar.selectbox("Retrieval mode", ["score", "faiss"])
query_for_faiss = st.sidebar.text_input("Semantic retrieval query (optional)", "")

st.sidebar.markdown("---")
st.sidebar.write("**Playbook file**: `playbook.jsonl`")
if st.sidebar.button("Reset Playbook"):
    if os.path.exists("playbook.jsonl"):
        os.remove("playbook.jsonl")
    st.success("Playbook reset.")

tab1, tab2, tab3, tab4 = st.tabs(["Run a Turn", "Playbook", "Visualizations", "About"])

with tab1:
    st.subheader("Run a Turn")
    user_query = st.text_area("Task / Query", "Create a 90‑minute workout playlist mixing genres; ensure ~5400s total.")
    colA, colB = st.columns(2)
    with colA:
        st.caption("Top‑K bullets injected into the Generator")
        topk = retriever_topk(k=k, mode=retrieval_mode, query=query_for_faiss)
        st.code(build_playbook_block(topk) or "(none yet)", language="markdown")
    with colB:
        st.caption("Environment")
        st.write("Model: `gpt-4.1-mini` (API key loaded from secrets/environment).")

    if st.button("Run Generator ➜ Reflector ➜ Curator"):
        try:
            g = generator(user_query, topk)
            st.success("Generator answer")
            st.write(g.get("answer",""))
            with st.expander("Trace"):
                st.json(g.get("trace", []))

            # Reflect
            bullets = reflector(user_query, g.get("answer",""), g.get("trace", []))
            st.info("Reflector bullets (deltas)")
            st.json(bullets)

            # Curate
            updated_topk = merge_deltas(bullets)

            st.success("Curator merged deltas. New Top‑K (score‑based):")
            st.json(updated_topk)

        except Exception as e:
            st.error(f"Run failed: {e}")

with tab2:
    st.subheader("Playbook View")
    st.caption("Raw bullets (score‑sorted)")
    st.json(get_topk_by_score(1000))

with tab3:
    st.subheader("Visualizations")

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

with tab4:
    st.markdown("""
**How it works**
1. **Retriever** gets Top‑K bullets from the JSON Playbook (score or FAISS).
2. **Generator** solves the task with Top‑K tips injected.
3. **Reflector** turns the run into new bullets.
4. **Curator** merges & dedups bullets, bumps helpful/harmful, updates the Playbook.

**Visualizations**
- Top bullets by score
- Helpful vs Harmful totals
- Tags frequency
- Daily timeline of playbook updates

> As requested: matplotlib only, one chart per figure, default colors.
""")
