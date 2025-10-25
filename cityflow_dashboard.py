import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime
import os
import pytz
import requests
from streamlit_autorefresh import st_autorefresh
from dotenv import load_dotenv
from agent_console import AgentConsole
from trend_predictor import forecast_congestion
from app.cityflow_agent import ACTIVE_MODEL



# ---------------------------------------------------
# ⚙️ ENV + CONFIG
# ---------------------------------------------------
load_dotenv()
API_BASE = os.getenv("API_BASE", "http://cityflow_api:8000")
DB_PATH = os.path.join(os.getcwd(), "cityflow.db")

st.set_page_config(
    page_title="CityFlow | London Traffic Dashboard",
    page_icon="🚗",
    layout="wide",
)
console = AgentConsole()

# ---------------------------------------------------
# 🎨 GLOBAL STYLING — CityFlow Console & Sidebar
# ---------------------------------------------------
st.markdown(
    """
    <style>
    /* --- Global --- */
    body, div, p, span {
        font-family: 'Inter', sans-serif !important;
    }

    /* --- Sidebar styling --- */
    section[data-testid="stSidebar"] {
        background-color: #0a0f1a !important;
        border-right: 1px solid #1e293b !important;
    }

    /* Sidebar header */
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h2 {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        border-bottom: 1px solid #1e293b !important;
        padding-bottom: 0.4rem !important;
    }

    /* --- Console Logs --- */
    .cityflow-log {
        background: linear-gradient(145deg, #0f172a 0%, #1e293b 100%);
        border-left: 4px solid #22c55e;
        border-radius: 10px;
        margin-bottom: 10px;
        padding: 10px 12px;
        box-shadow: 0 0 12px rgba(34, 197, 94, 0.2);
        color: #e2e8f0;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.88rem;
    }

    .cityflow-log.error {
        border-left: 4px solid #ef4444;
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.25);
    }

    .cityflow-log .time {
        color: #94a3b8;
        font-size: 0.75rem;
    }

    /* --- Chat input field --- */
    div[data-testid="stChatInput"] textarea {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
        border: 1px solid #334155 !important;
    }

    /* --- Buttons --- */
    button[kind="secondary"] {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
    }

    /* --- Scrollbar --- */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------
# 🕒 TIMEZONE
# ---------------------------------------------------
london_tz = pytz.timezone("Europe/London")
current_time = datetime.now(london_tz)
REFRESH_INTERVAL = 15 * 60

# ---------------------------------------------------
# 📥 LOAD DATA
# ---------------------------------------------------
@st.cache_data(ttl=900)
def load_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM road_status", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

df = load_data()

# ---------------------------------------------------
# 🧠 FETCH AI SUMMARY
# ---------------------------------------------------
try:
    res = requests.get(f"{API_BASE}/summary", timeout=10)
    ai_summary = res.json().get("summary", "⚠️ No AI summary available.")
    ai_status, ai_color = ("🧠 AI Summary Ready", "blue")
except Exception as e:
    ai_summary = f"⚠️ Error fetching summary: {e}"
    ai_status, ai_color = ("🔴 Summary Unavailable", "red")

# ---------------------------------------------------
# 🚦 FEED STATUS
# ---------------------------------------------------
if not df.empty:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    latest_time = df["timestamp"].max()
    latest_uk = (
        latest_time.tz_localize("UTC").astimezone(london_tz)
        if latest_time.tzinfo is None
        else latest_time.astimezone(london_tz)
    )
    minutes_diff = (current_time - latest_uk).total_seconds() / 60
    if minutes_diff < 15:
        feed_status, feed_color = "🟢 TfL Feed Active", "green"
    elif minutes_diff < 60:
        feed_status, feed_color = "🟠 TfL Feed Slightly Stale", "orange"
    else:
        feed_status, feed_color = "🔴 TfL Feed Outdated", "red"
else:
    feed_status, feed_color = "🔴 TfL Feed Down", "red"

# ---------------------------------------------------
# 🧭 HEADER
# ---------------------------------------------------
st.markdown(f"""
<div style="background:#0f172a;color:white;padding:1rem;border-radius:10px;
display:flex;justify-content:space-between;align-items:center;">
  <div style="font-size:1.2rem;font-weight:600;">🚦 CityFlow – Live TfL Traffic</div>
  <div>
    <span style="background:#1e293b;padding:0.5rem;border-radius:6px;">{feed_status}</span>
    <span style="margin-left:10px;background:#1e293b;padding:0.5rem;border-radius:6px;">{ai_status}</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.caption(f"Updated {current_time.strftime('%Y-%m-%d %H:%M')} (London time)")
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="data_refresh")

# ---------------------------------------------------
# 🧠 ACTIVE MODEL INDICATOR
# ---------------------------------------------------

# 🎨 Colour-coded active-model badge
if ACTIVE_MODEL:
    if "70b" in ACTIVE_MODEL:
        color = "#22c55e"      # green → primary
    elif "8b" in ACTIVE_MODEL:
        color = "#f97316"      # orange → fallback
    else:
        color = "#ef4444"      # red → unknown/offline

    st.markdown(
        f"""
        <div style="padding:6px 10px;border-radius:8px;
                    background-color:{color};color:white;
                    display:inline-block;font-weight:600;
                    margin-bottom:1rem;">
            🧠 Active Model: {ACTIVE_MODEL}
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div style="padding:6px 10px;border-radius:8px;
                    background-color:#9ca3af;color:white;
                    display:inline-block;font-weight:600;
                    margin-bottom:1rem;">
            🧠 Active Model: loading …
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------
# 🔗 BACKEND CONNECTION INDICATOR
# ---------------------------------------------------
try:
    ping = requests.get(f"{API_BASE}/health", timeout=5)
    status = ping.json().get("status", "")
    if status == "ok":
        backend_status = "🟢 Backend Connected"
        backend_color = "#16a34a"
    else:
        backend_status = "🟠 Backend Initializing"
        backend_color = "#f59e0b"
except Exception:
    backend_status = "🔴 Backend Unreachable"
    backend_color = "#dc2626"

st.markdown(
    f"""
    <div style='background-color:#0f172a;
                color:white;
                padding:0.7rem 1rem;
                border-radius:8px;
                font-size:0.9rem;
                margin-bottom:1rem;
                display:flex;
                justify-content:space-between;
                align-items:center;'>
        <div>{backend_status}</div>
        <div style='color:#94a3b8;'>Connected to: <b>{API_BASE}</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# 📊 SEVERITY BREAKDOWN
# ---------------------------------------------------
if not df.empty:
    st.subheader("📊 Traffic Severity Breakdown")
    severity_count = df["severity"].value_counts(dropna=False).reset_index()
    severity_count.columns = ["Severity", "Count"]
    fig = px.bar(
        severity_count,
        x="Severity",
        y="Count",
        text="Count",
        color="Severity",
        color_discrete_map={
            "Serious": "#dc2626",
            "Moderate": "#f59e0b",
            "Minimal": "#16a34a",
        },
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, plot_bgcolor="#f8fafc")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("⚠️ No TfL data found. Run `tfl_pipeline.py` first.")

# ---------------------------------------------------
# 📈 DAILY TREND + FORECAST
# ---------------------------------------------------
st.subheader("📈 Daily Disruption Trend & Forecast")

try:
    conn = sqlite3.connect(DB_PATH)
    trend_df = pd.read_sql("SELECT * FROM daily_trends ORDER BY date", conn)
    conn.close()
except Exception:
    trend_df = pd.DataFrame()

if len(trend_df) > 1:
    fig = px.line(
        trend_df,
        x="date",
        y="disruptions",
        markers=True,
        color_discrete_sequence=["#2563eb"],
        title="Disruptions Over Time (Daily)",
    )
    st.plotly_chart(fig, use_container_width=True)

    _, forecast_text = forecast_congestion()
    st.success(f"📈 {forecast_text}")
else:
    st.info("No daily trend data yet — will populate after first refresh.")

# ---------------------------------------------------
# 🧠 AI SUMMARY
# ---------------------------------------------------
st.subheader("🧠 AI Traffic Summary")

with st.expander("📜 View AI-Generated Summary", expanded=True):
    st.markdown(
        f"<div style='background:#f8fafc;padding:1rem;border-radius:10px;'>{ai_summary}</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------
# 💬 CITYFLOW CHAT AGENT
# ---------------------------------------------------
st.markdown("---")
st.subheader("💬 CityFlow AI Traffic Assistant")

if st.button("🔄 Reset Conversation"):
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask about London traffic..."):
    # 🧠 Start a new console session (no fixed tool name)
    console.start("CityFlow Agent")
    with st.spinner("🤖 Thinking..."):
        answer = "⚠️ No response received."
        steps = []
        tool_used = "LLM Query"

        try:
            # Send query to backend
            payload = {"input": str(user_query).strip()}
            res = requests.post(f"{API_BASE}/ask", json=payload, timeout=90)
            data = res.json()

            # Extract response data
            answer = data.get("answer", "⚠️ No answer generated.")
            steps = data.get("intermediate_steps", [])
            tool_used = data.get("tool_used", "LLM Query")

            # Save tool name to session + console
            st.session_state["last_tool_used"] = tool_used
            console.end(f"✅ {tool_used}")  # ✅ Dynamic tool name now shown

        except requests.exceptions.ReadTimeout:
            answer = "⚠️ The request to the backend timed out (30s). Please try again shortly."
            tool_used = "LLM Query"
            console.end("❌ Timeout")

        except Exception as e:
            answer = f"⚠️ API Error: {e}"
            tool_used = "LLM Query"
            console.end("❌ Failed")

    # --- Display assistant response ---
    st.chat_message("assistant").markdown(answer)

    # --- Tool tag with emoji color map ---
    tool_color_map = {
        "fetch": ("#f97316", "📡"),
        "summarize": ("#3b82f6", "🧠"),
        "analyze": ("#a855f7", "📊"),
        "compare": ("#a855f7", "📊"),
        "respond": ("#22c55e", "💬"),
        "recommend": ("#eab308", "🗺️"),
    }

    color, emoji = next(
        ((c, e) for key, (c, e) in tool_color_map.items() if key in tool_used.lower()),
        ("#94a3b8", "⚙️"),
    )

    st.markdown(
        f"""
        <div style='margin-top:-8px;margin-bottom:8px;
                    font-size:0.9rem;
                    color:white;
                    background-color:{color};
                    padding:4px 10px;
                    border-radius:6px;
                    display:inline-block;'>
            {emoji} <b>{tool_used}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Save chat history ---
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # --- Thought process trace ---
    if steps:
        with st.expander("🧩 Agent Thought Process", expanded=False):
            for i, step in enumerate(steps):
                thought = step[0] if len(step) > 0 else ""
                action = str(step[1]) if len(step) > 1 else ""
                st.markdown(
                    f"""
                    <div style="padding:8px;margin-bottom:6px;
                                border-left:4px solid #3b82f6;
                                background-color:#0f172a;border-radius:6px;
                                font-family:'Inter',sans-serif;">
                        <b>🧠 Thought {i+1}</b><br>
                        <span style='color:#e2e8f0;font-size:0.9rem;'>{thought}</span><br>
                        <b>⚙️ Action:</b> 
                        <span style='color:#60a5fa;'>{action}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------
# 🧠 SIDEBAR CONSOLE
# ---------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 Agent Console Log")

    # Ensure session_state has console logs
    if "console_logs" not in st.session_state:
        st.session_state.console_logs = []

    # Merge any new logs from AgentConsole
    if hasattr(console, "logs") and console.logs:
        st.session_state.console_logs.extend(console.logs)
        console.logs.clear()  # Prevent duplicates

    logs_to_display = st.session_state.console_logs[-5:]  # last 5 entries

    if logs_to_display:
        for log in reversed(logs_to_display):
            color = "#16a34a" if "Success" in log["status"] else "#dc2626"
            st.markdown(
                f"""
                <div style='padding:10px 12px;margin-bottom:8px;
                            background:#0f172a;border-radius:8px;
                            border-left:5px solid {color};
                            font-family:JetBrains Mono, monospace;
                            color:#e2e8f0;font-size:0.9rem;'>
                    <b>{log['tool']}</b> — {log['status']} ({log['time']}s)<br>
                    <span style='color:#94a3b8;font-size:0.8rem;'>
                        Last tool: {st.session_state.get('last_tool_used', 'LLM Query')}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption("🕒 Console empty — Ask the AI agent to begin.")

# ---------------------------------------------------
# 📅 FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption(
    f"🕒 Updated on {datetime.now(london_tz).strftime('%Y-%m-%d %H:%M')} (London time)"
)
st.caption("🚗 Powered by TfL Open Data • Built with ❤️ by Aditya Iyer • CityFlow © 2025")
