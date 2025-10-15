import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime
import os
from streamlit_autorefresh import st_autorefresh
import pytz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from dotenv import load_dotenv
import requests

# ---------------------------------------
# ⚙️  Load environment variables
# ---------------------------------------
load_dotenv(dotenv_path=".env")

# ---------------------------------------
# 🚦 Page Config
# ---------------------------------------
st.set_page_config(
    page_title="CityFlow | London Traffic Dashboard",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------------------
# 📥 Load SQLite Data
# ---------------------------------------
@st.cache_data(ttl=900)
def load_data():
    conn = sqlite3.connect("cityflow.db")
    df = pd.read_sql("SELECT * FROM road_status", conn)
    conn.close()
    return df

df = load_data()

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ---------------------------------------
# 🕒 Timezone Setup
# ---------------------------------------
london_tz = pytz.timezone("Europe/London")
current_time = datetime.now(london_tz)

# ---------------------------------------
# 🧠 AI Summary File Check
# ---------------------------------------
summary_file = "latest_summary.txt"
if os.path.exists(summary_file):
    with open(summary_file, "r") as f:
        ai_summary = f.read().strip()
    ai_status, ai_color = (
        ("🧠 AI Summary Ready", "blue") if ai_summary else ("🟠 Summary Empty", "orange")
    )
else:
    ai_summary, ai_status, ai_color = "", "⚪ Awaiting Summary", "gray"

# ---------------------------------------
# 🚦 TfL Feed Freshness
# ---------------------------------------
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

# ---------------------------------------
# 💅 Header CSS
# ---------------------------------------
st.markdown("""
<style>
.status-bar {
  display:flex; justify-content:space-between; align-items:center;
  background-color:#0f172a; color:white;
  padding:1rem 1.2rem; border-radius:10px; margin-bottom:15px;
  font-family:'Inter',sans-serif;
}
.badge {
  display:inline-block; padding:0.35rem 0.7rem;
  border-radius:6px; font-size:0.9rem; font-weight:600; margin-left:0.4rem;
}
.green{background:#16a34a;color:white;}
.orange{background:#f97316;color:white;}
.red{background:#dc2626;color:white;}
.gray{background:#374151;color:#d1d5db;}
.blue{background:#2563eb;color:white;}
a{color:white;text-decoration:none;} a:hover{text-decoration:underline;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# ⏳ Countdown Logic
# ---------------------------------------
REFRESH_INTERVAL = 15 * 60
if "refresh_start" not in st.session_state:
    st.session_state.refresh_start = datetime.now()

elapsed = (datetime.now() - st.session_state.refresh_start).total_seconds()
remaining = max(0, REFRESH_INTERVAL - int(elapsed))
if remaining == 0:
    st.session_state.refresh_start = datetime.now()
    remaining = REFRESH_INTERVAL

# ---------------------------------------
# 🧭 Header
# ---------------------------------------
refresh_label = f"🔁 Auto-refresh • {current_time.strftime('%Y-%m-%d %H:%M')} (London)"
st.markdown(f"""
<div class="status-bar">
  <div style="font-size:1.3rem; font-weight:600;">🚦 CityFlow – Live TfL Traffic Disruptions</div>
  <div>
    <span class="badge {feed_color}">{feed_status}</span>
    <a href="#ai-summary">
      <span class="badge {ai_color}" title="View AI-generated summary below">{ai_status}</span>
    </a>
    <span class="badge gray" id="refresh-timer">
      {refresh_label}<br>
      <small>⏳ Next refresh in <span id="countdown">{remaining//60:02d}:{remaining%60:02d}</span></small>
    </span>
  </div>
</div>

<script>
let remaining = {remaining};
function fmt(sec) {{
  let m=Math.floor(sec/60), s=sec%60;
  return m.toString().padStart(2,'0')+":"+s.toString().padStart(2,'0');
}}
function tick(){{
  remaining--;
  if(remaining<0) remaining={REFRESH_INTERVAL};
  let color="#10b981";
  if(remaining<60) color="#dc2626";
  else if(remaining<180) color="#f97316";
  document.getElementById("countdown").style.color=color;
  document.getElementById("countdown").innerText=fmt(remaining);
}}
setInterval(tick,1000);
</script>
""", unsafe_allow_html=True)

st.caption("AI-powered London traffic insights • Updated automatically every 15 minutes.")
st.markdown("<a id='ai-summary'></a>", unsafe_allow_html=True)

# ---------------------------------------
# 📊 Traffic Severity Breakdown
# ---------------------------------------
if not df.empty:
    st.subheader("📊 Traffic Severity Breakdown")
    severity_count = df["severity"].value_counts(dropna=False).reset_index()
    severity_count.columns = ["severity", "count"]

    fig = px.bar(
        severity_count,
        x="severity", y="count", color="severity", text="count",
        color_discrete_map={"Serious": "#FF4B4B", "Moderate": "#FFA500", "Minimal": "#00C853"},
        title="Traffic Severity Breakdown by Level"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("⚠️ No data available. Run `tfl_pipeline.py` first!")

# ---------------------------------------
# 📈 Trend Over Time
# ---------------------------------------
if not df.empty:
    st.subheader("📈 Disruption Trend Over Time")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    trend = df.groupby([pd.Grouper(key="timestamp", freq="H"), "severity"]).size().reset_index(name="count")
    fig = px.line(
        trend, x="timestamp", y="count", color="severity", markers=True,
        color_discrete_map={"Serious":"#FF4B4B","Moderate":"#FFA500","Minimal":"#00C853"},
        title="Traffic Disruptions Over Time (Hourly)"
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------
# 🧠 AI Summary (Expander)
# ---------------------------------------
st.subheader("🧠 AI-Generated Traffic Summary")
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="data_refresh")
if ai_summary:
    expanded = feed_color == "green"
    with st.expander("📜 Click to view AI Traffic Summary", expanded=expanded):
        st.markdown(f"<div style='background-color:#f1f5f9; padding:15px; border-radius:10px;'><p style='font-size:1rem; color:#111827; line-height:1.6;'>{ai_summary}</p></div>", unsafe_allow_html=True)
else:
    st.info("📭 No AI summary yet. Run `tfl_pipeline.py` to generate one.")

# ---------------------------------------
# 💬 CityFlow AI Traffic Assistant (Smart Multi-Query)
# ---------------------------------------
st.markdown("---")
st.subheader("💬 CityFlow AI Traffic Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Borough / area mapping for detection
BOROUGH_MAP = {
    "camden": ["camden", "kentish town", "holborn"],
    "westminster": ["westminster", "paddington", "victoria", "mayfair"],
    "southwark": ["southwark", "elephant", "bermondsey"],
    "islington": ["islington", "angel"],
    "hackney": ["hackney", "dalston", "shoreditch"],
    "greenwich": ["greenwich", "blackheath", "woolwich"],
    "kensington and chelsea": ["kensington", "chelsea", "earls court"],
    "lambeth": ["lambeth", "brixton", "clapham"],
    "tower hamlets": ["tower hamlets", "whitechapel", "canary wharf"],
    "lewisham": ["lewisham", "catford"],
    "croydon": ["croydon", "purley"],
    "ealing": ["ealing", "acton"],
    "brent": ["brent", "wembley"],
    "bromley": ["bromley", "orpington"],
    "newham": ["newham", "stratford"],
    "barking and dagenham": ["barking", "dagenham"],
    "redbridge": ["ilford", "wanstead"],
    "waltham forest": ["walthamstow", "leyton"],
    "hounslow": ["hounslow", "chiswick"],
    "richmond upon thames": ["richmond", "twickenham"],
    "merton": ["wimbledon", "morden"],
    "sutton": ["sutton", "carshalton"],
    "kingston upon thames": ["kingston", "surbiton"],
    "havering": ["romford", "upminster"],
}

def detect_boroughs(query):
    query = query.lower()
    detected = []
    for borough, keywords in BOROUGH_MAP.items():
        if any(k in query for k in keywords):
            detected.append(borough)
    return list(set(detected))

def detect_intent(query):
    query = query.lower()
    if any(k in query for k in ["closure", "closed", "blocked", "restriction"]):
        return "closure"
    elif any(k in query for k in ["delay", "slow", "traffic", "congestion"]):
        return "delay"
    elif any(k in query for k in ["clear", "open", "resolved"]):
        return "clear"
    else:
        return "general"

if user_query := st.chat_input("Ask about London traffic disruptions..."):
    st.session_state.messages.append({"role": "user", "content": user_query})

    conn = sqlite3.connect("cityflow.db")
    df_context = pd.read_sql("SELECT * FROM road_status", conn)
    conn.close()

    detected_boroughs = detect_boroughs(user_query)
    intent = detect_intent(user_query)

    if detected_boroughs:
        st.info(f"📍 Detected areas: {', '.join([b.title() for b in detected_boroughs])}")
    else:
        st.caption("⚙️ No specific borough detected — showing all London disruptions.")

    if detected_boroughs:
        mask = df_context.apply(
            lambda r: any(k in str(r).lower() for b in detected_boroughs for k in BOROUGH_MAP[b]),
            axis=1,
        )
        filtered = df_context[mask]
    else:
        filtered = df_context

    if intent == "closure":
        filtered = filtered[filtered["comments"].str.contains("clos", case=False, na=False)]
    elif intent == "delay":
        filtered = filtered[filtered["comments"].str.contains("delay|congestion", case=False, na=False)]

    filtered["text"] = filtered.apply(
        lambda r: f"{r['road']} — {r['severity']} — {r['category']} — {r['comments']}", axis=1
    )

    if filtered.empty:
        answer = "⚠️ No disruptions matching your query found in TfL data."
    else:
        if "embeddings" not in st.session_state or len(st.session_state.embeddings) != len(filtered):
            with st.spinner("🔍 Encoding live traffic data..."):
                st.session_state.embeddings = embedder.encode(
                    filtered["text"].tolist(), show_progress_bar=False
                )

        q_vec = embedder.encode([user_query])
        sims = cosine_similarity(q_vec, st.session_state.embeddings)[0]
        top_k_idx = np.argsort(sims)[-5:][::-1]
        top_contexts = [filtered.iloc[i]["text"] for i in top_k_idx]
        context_text = "\n".join(top_contexts)

        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are CityFlow, an AI trained on live TfL data. Answer clearly, concisely, and reference boroughs when relevant."},
                {"role": "user", "content": f"Context:\n{context_text}\n\nUser query: {user_query}"}
            ],
            "max_tokens": 350,
        }

        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json=payload,
            )
            if resp.status_code == 200:
                answer = resp.json()["choices"][0]["message"]["content"].strip()
            else:
                answer = f"⚠️ API error ({resp.status_code})"
        except Exception as e:
            answer = f"⚠️ Request failed: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

# ---------------------------------------
# 📅 Footer
# ---------------------------------------
st.markdown("---")
st.caption(f"🕒 Updated on {datetime.now(london_tz).strftime('%Y-%m-%d %H:%M')} (London time)")
st.caption("🚗 Powered by TfL Open Data • Built with ❤️ by Aditya Iyer • CityFlow © 2025")
