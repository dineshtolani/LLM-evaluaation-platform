import streamlit as st
import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

import os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8080") + "/api"

st.set_page_config(page_title="LLM Evaluation Dashboard", layout="wide")
st.title("🤖 LLM Evaluation & Monitoring Platform")


@st.cache_data(ttl=10)
def fetch_json(endpoint):
    try:
        resp = httpx.get(f"{API_BASE}/{endpoint}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def post_json(endpoint, data):
    try:
        resp = httpx.post(f"{API_BASE}/{endpoint}", json=data, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "📝 Evaluate", "📋 History", "⚙️ Prompts", "⚠️ Toxicity Report", "🧠 Hallucination Report"])

with tab1:
    st.subheader("Platform Overview")
    stats = fetch_json("evaluations/stats")
    if stats:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Evaluations", stats.get("total_evaluations", 0))
        col2.metric("Avg Latency", f"{stats.get('avg_latency_ms', 0):.0f}ms")
        col3.metric("Hallucination", f"{stats.get('avg_hallucination_score', 0):.2%}")
        col4.metric("Quality", f"{stats.get('avg_quality_score', 0):.2%}")
        col5.metric("Toxicity", f"{stats.get('avg_toxicity_score', 0):.2%}")

        st.metric("Total Tokens Consumed", f"{stats.get('total_tokens_consumed', 0):,}")

    evals = fetch_json("evaluations?page_size=50")
    if evals and evals.get("items"):
        df = pd.DataFrame(evals["items"])
        st.subheader("Latency Trend")
        fig = px.line(df, x="created_at", y="latency_ms", title="Response Time (ms)")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.scatter(df, x="hallucination_score", y="quality_score",
                            hover_data=["model_name"], title="Hallucination vs Quality")
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            cost_df = df[df["token_cost"] > 0]
            fig3 = px.bar(cost_df, x="created_at", y="token_cost", title="Cost per Evaluation ($)")
            st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.subheader("Run Evaluation")

    eval_mode = st.radio("Prompt source", ["✏️ Quick Eval (any text)", "📂 Saved prompt"], horizontal=True)

    prompts = fetch_json("prompts?page_size=100")
    models = fetch_json("models?page_size=100")

    model_options = {}
    if models and models.get("items"):
        for m in models["items"]:
            model_options[m["name"]] = m["name"]
    if not model_options:
        model_options["tinyllama"] = "tinyllama"

    with st.container():
        if eval_mode.startswith("✏️"):
            prompt_text = st.text_area("Enter your prompt", height=120,
                                       placeholder="e.g., What is the capital of France?")
            expected = st.text_area("Expected output (optional)", height=60,
                                    placeholder="e.g., Paris")
            selected_model = st.selectbox("Model", options=list(model_options.keys()))
            col_t, col_m = st.columns(2)
            with col_t:
                temp = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
            with col_m:
                max_tokens = st.slider("Max Tokens", 50, 2000, 500, 50)

            if st.button("🚀 Quick Evaluate", type="primary"):
                if not prompt_text.strip():
                    st.warning("Enter a prompt first.")
                else:
                    with st.spinner("Creating prompt + running evaluation..."):
                        created = post_json("prompts", {
                            "name": f"quick-{datetime.now().strftime('%H%M%S')}",
                            "content": prompt_text,
                            "category": "other",
                            "expected_output": expected if expected.strip() else None,
                        })
                    if created:
                        with st.spinner("Running evaluation..."):
                            result = post_json("evaluations", {
                                "prompt_id": created["id"],
                                "model_name": selected_model,
                                "temperature": temp,
                                "max_tokens": max_tokens,
                            })
                        if result:
                            st.success("Evaluation complete!")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Latency", f"{result['latency_ms']:.0f}ms")
                            c2.metric("Tokens", result['total_tokens'])
                            c3.metric("Cost", f"${result['token_cost']:.6f}")
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Hallucination", f"{result.get('hallucination_score', 0):.2%}")
                            c2.metric("Quality", f"{result.get('quality_score', 0):.2%}")
                            c3.metric("Relevance", f"{result.get('relevance_score', 0):.2%}")
                            c4.metric("Toxicity", f"{result.get('toxicity_score', 0):.2%}")
                            if result.get("is_toxic"):
                                st.warning("⚠️ Toxic content detected")
                            st.text_area("Response", result["response"], height=200)
        else:
            prompt_options = {}
            if prompts and prompts.get("items"):
                for p in prompts["items"]:
                    prompt_options[f"{p['name']} ({p['id'][:8]})"] = p["id"]

            col1, col2 = st.columns(2)
            with col1:
                selected_prompt = st.selectbox("Select Prompt", options=list(prompt_options.keys()))
            with col2:
                selected_model = st.selectbox("Select Model", options=list(model_options.keys()))

            temp = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
            max_tokens = st.slider("Max Tokens", 50, 2000, 500, 50)

            if st.button("🚀 Run Evaluation", type="primary"):
                with st.spinner("Running evaluation..."):
                    prompt_id = prompt_options[selected_prompt]
                    result = post_json("evaluations", {
                        "prompt_id": prompt_id,
                        "model_name": selected_model,
                        "temperature": temp,
                        "max_tokens": max_tokens,
                    })
                if result:
                    st.success("Evaluation complete!")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Latency", f"{result['latency_ms']:.0f}ms")
                    col2.metric("Tokens", result['total_tokens'])
                    col3.metric("Cost", f"${result['token_cost']:.6f}")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Hallucination", f"{result.get('hallucination_score', 0):.2%}")
                    col2.metric("Quality", f"{result.get('quality_score', 0):.2%}")
                    col3.metric("Relevance", f"{result.get('relevance_score', 0):.2%}")
                    col4.metric("Toxicity", f"{result.get('toxicity_score', 0):.2%}")
                    st.text_area("Response", result["response"], height=200)

with tab3:
    st.subheader("Evaluation History")
    evals = fetch_json("evaluations?page_size=100")
    if evals and evals.get("items"):
        df = pd.DataFrame(evals["items"])
        cols = ["created_at", "model_name", "prompt_name", "latency_ms", "total_tokens",
                "hallucination_score", "quality_score", "toxicity_score", "token_cost"]
        display_cols = [c for c in cols if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)

    st.subheader("Health Check")
    health = fetch_json("../health")
    if health:
        st.json(health)

with tab4:
    st.subheader("Manage Prompts")
    prompts = fetch_json("prompts?page_size=100")
    if prompts and prompts.get("items"):
        for p in prompts["items"]:
            with st.expander(f"📄 {p['name']} (v{p['version']})"):
                st.text(f"Category: {p['category']} | Status: {p['status']}")
                st.text_area("Prompt", p['content'], height=100, key=f"content_{p['id']}")
                if p.get("expected_output"):
                    st.text_area("Expected Output", p['expected_output'], height=60, key=f"expected_{p['id']}")
                st.caption(f"Created: {p['created_at']}")

with tab5:
    st.subheader("⚠️ Toxicity Report")
    report = fetch_json("evaluations/toxicity-report")
    if report:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Evaluations", report["total_evaluations"])
        col2.metric("Toxic Flagged", f'{report["toxic_count"]} ({report["toxic_percentage"]}%)')
        col3.metric("Avg Toxicity", f'{report["avg_toxicity"]:.2%}')
        col4.metric("Max Toxicity", f'{report["max_toxicity"]:.2%}')

        if report["toxic_by_category"]:
            st.subheader("Toxicity by Category")
            cat_df = pd.DataFrame([
                {"Category": cat, "Count": cnt}
                for cat, cnt in report["toxic_by_category"].items()
            ])
            st.bar_chart(cat_df.set_index("Category"))

        if report["most_toxic"]:
            st.subheader("Most Toxic Responses")
            for item in report["most_toxic"]:
                with st.expander(f"Score: {item['toxicity_score']:.2%} | {item['prompt_name']} | {item['model_name']}"):
                    st.text(f"Response: {item['response_preview']}...")
                    st.text(f"Date: {item['created_at']}")
        else:
            st.success("No toxic content detected in any evaluation.")

with tab6:
    st.subheader("🧠 Hallucination Report")
    report = fetch_json("evaluations/hallucination-report")
    if report:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Evaluations", report["total_evaluations"])
        col2.metric("Avg Hallucination", f'{report["avg_hallucination_score"]:.2%}')
        col3.metric("Max Hallucination", f'{report["max_hallucination_score"]:.2%}')
        col4.metric("Min Hallucination", f'{report["min_hallucination_score"]:.2%}')

        col1, col2, col3 = st.columns(3)
        col1.metric("High Hallucination Count", report["high_hallucination_count"])
        col2.metric("High Hallucination %", f'{report["high_hallucination_percentage"]}%')
        col3.metric("Threshold", f'{report["high_hallucination_threshold"]:.0%}')

        if report["most_hallucinated"]:
            st.subheader("Most Hallucinated Responses")
            for item in report["most_hallucinated"]:
                label = f"Hallucination: {item['hallucination_score']:.0%} | {item['prompt_name']} | {item['model_name']} | Quality: {item['quality_score']:.0%}"
                with st.expander(label):
                    st.text(f"Prompt: {item.get('prompt_content', 'N/A')}")
                    st.text(f"Response: {item['response_preview']}...")
                    st.text(f"Latency: {item['latency_ms']}ms | Tokens: {item['tokens']} | Date: {item['created_at']}")
        else:
            st.success("No evaluations with hallucination scores yet.")
