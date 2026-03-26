import streamlit as st
import time
import pandas as pd

from core import (
    full_system,
    compute_dashboard_metrics,
    get_logs_df,
    get_cost_comparison_df,
    compute_efficiency_score
)

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Adaptive LLM System", layout="wide")

# ===== STATE =====
if "messages" not in st.session_state:
    st.session_state.messages = []

if "logs" not in st.session_state:
    st.session_state.logs = []

# ===== SIDEBAR =====
with st.sidebar:
    st.title("⚙️ Controls")

    dashboard_mode = st.toggle("📊 Dashboard Mode")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.session_state.logs = []
        st.rerun()

    st.markdown("---")

    # ===== EXAMPLE QUERIES =====
    st.markdown("## 💡 Example Queries")

    examples = [
        "What is machine learning?",
        "Explain transformers architecture",
        "Write Python code for binary search",
        "Design a scalable chat system",
        "Difference between CNN and RNN"
    ]

    selected_example = st.selectbox("Try an example", [""] + examples)

# =========================
# 📊 DASHBOARD MODE
# =========================
if dashboard_mode:

    st.title("📊 System Dashboard")

    metrics = compute_dashboard_metrics()
    df = get_logs_df()

    if metrics is None:
        st.warning("Run some queries first.")
        st.stop()

    # ===== METRICS =====
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Queries", metrics["total_queries"])
    c2.metric("Avg Latency", f"{metrics['avg_latency']:.2f}s")
    c3.metric("Avg Cost", f"${metrics['avg_cost']:.5f}")
    c4.metric("Efficiency", f"{metrics['avg_efficiency']:.2f}")

    st.success(f"💰 Cost Reduction: {metrics['cost_reduction']:.2f}%")

    # ===== COST GRAPH =====
    st.markdown("## 📉 Cost Comparison (Cumulative)")

    cost_df = get_cost_comparison_df()

    if cost_df is not None:
        st.line_chart(cost_df)

        savings = cost_df["Baseline Cost"].iloc[-1] - cost_df["Optimized Cost"].iloc[-1]
        st.metric("💰 Total Savings", f"${savings:.4f}")
    else:
        st.warning("Not enough data")

    # ===== EFFICIENCY GRAPH =====
    st.markdown("## ⚡ Efficiency Over Time")

    df["efficiency_score"] = df.apply(compute_efficiency_score, axis=1)
    st.line_chart(df["efficiency_score"])

    # ===== MODEL USAGE =====
    st.markdown("## 🤖 Model Usage")
    st.bar_chart(df["model_type"].value_counts())

    # ===== LATENCY VS COST =====
    st.markdown("## ⚡ Latency vs Cost")
    st.scatter_chart(df[["latency", "cost"]])

    # ===== LOGS =====
    st.markdown("## 📋 Logs")
    st.dataframe(df)

    st.stop()

# =========================
# 💬 CHAT UI
# =========================

# ===== HEADER =====
st.markdown("<h1 style='text-align:center;'>🤖 Adaptive LLM Chat</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Smart routing • Cost-aware • Multi-model AI</p>", unsafe_allow_html=True)

# ===== ABOUT SECTION =====
with st.expander("ℹ️ About This System"):
    st.markdown("""
### 🚀 Adaptive Multi-Model LLM Router

This system intelligently routes user queries to different AI models based on complexity.

---

### 🧠 How it works

1. **Query Classification**
   - Uses a trained DistilBERT model
   - Classifies queries into levels (L1 → L5)

2. **Dynamic Routing**
   - Easy Queries → Small model
   - Noramal Queries → Medium model
   - Complex Queries → Large model

3. **Cost Optimization**
   - Reduces unnecessary usage of expensive models
   - Improves efficiency

4. **Analytics**
   - Tracks cost, latency, efficiency
   - Compares with baseline system

---

### 🎯 Goal

Build a **Green AI system** that reduces cost while maintaining performance.
""")

# ===== DISPLAY CHAT =====
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div style='background:#f1f5f9;padding:12px;border-radius:10px;"
            f"margin-bottom:10px;text-align:right;max-width:70%;margin-left:auto;'>"
            f"{msg['content']}</div>",
            unsafe_allow_html=True
        )
    else:
        with st.container():
            st.markdown(
                "<div style='background:#ffffff;padding:14px;border-radius:10px;"
                "border:1px solid #e2e8f0;max-width:70%;margin-bottom:10px;'>",
                unsafe_allow_html=True
            )

            st.markdown(msg["content"])
            st.markdown("</div>", unsafe_allow_html=True)

            st.caption(
                f"Model: {msg['model_type']} | {msg['actual_model']} | "
                f"{msg['latency']:.2f}s | ${msg['cost']:.5f}"
            )

# ===== INPUT =====
prompt = st.chat_input("Ask anything...")

# Inject example query
if selected_example and not prompt:
    prompt = selected_example

if prompt:
    # Save user
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    st.markdown(
        f"<div style='background:#f1f5f9;padding:12px;border-radius:10px;"
        f"text-align:right;max-width:70%;margin-left:auto;'>"
        f"{prompt}</div>",
        unsafe_allow_html=True
    )

    # ===== MODEL CALL =====
    with st.spinner("Routing..."):
        response, model_type, actual_model, latency, cost, log = full_system(prompt)

    st.session_state.logs.append(log)

    # ===== STREAMING =====
    placeholder = st.empty()
    full_text = ""

    for word in response.split(" "):
        full_text += word + " "
        with placeholder.container():
            st.markdown(
                "<div style='background:#ffffff;padding:14px;border-radius:10px;"
                "border:1px solid #e2e8f0;max-width:70%;'>",
                unsafe_allow_html=True
            )
            st.markdown(full_text)
            st.markdown("</div>", unsafe_allow_html=True)
        time.sleep(0.01)

    # FINAL RENDER
    with placeholder.container():
        st.markdown(
            "<div style='background:#ffffff;padding:14px;border-radius:10px;"
            "border:1px solid #e2e8f0;max-width:70%;'>",
            unsafe_allow_html=True
        )
        st.markdown(response)
        st.markdown("</div>", unsafe_allow_html=True)

    # META
    st.caption(
        f"Model: {model_type.upper()} | {actual_model} | "
        f"{latency:.2f}s | ${cost:.5f}"
    )

    # Save assistant
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "model_type": model_type.upper(),
        "actual_model": actual_model,
        "latency": latency,
        "cost": cost
    })