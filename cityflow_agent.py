# cityflow_agent.py
import os
import sqlite3
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# ----------------------------
# 🔑 Setup & Config
# ----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
london_tz = pytz.timezone("Europe/London")

# Initialize LLM (Groq endpoint)
llm = ChatOpenAI(
    model_name="llama-3.3-70b-versatile",
    openai_api_key=GROQ_API_KEY,
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0.4
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# 🛰 Tool 1: Fetch latest TfL data
# ----------------------------
@tool("fetch_tfl_data")
def fetch_tfl_data() -> str:
    """Fetch the latest TfL disruption data from the local SQLite database."""
    conn = sqlite3.connect("cityflow.db")
    df = pd.read_sql("SELECT * FROM road_status", conn)
    conn.close()
    latest = df["timestamp"].max() if not df.empty else "No data"
    return f"Fetched {len(df)} disruptions. Latest update: {latest}"

# ----------------------------
# 🧩 Tool 2: Summarize trends
# ----------------------------
@tool("summarize_trends")
def summarize_trends() -> str:
    """Analyze recent disruptions and summarize key traffic trends."""
    conn = sqlite3.connect("cityflow.db")
    df = pd.read_sql("SELECT * FROM road_status", conn)
    conn.close()

    if df.empty:
        return "No recent TfL data available."

    top_roads = df["road"].value_counts().head(3).index.tolist()
    severe_count = (df["severity"] == "Serious").sum()

    context = f"""
    Total disruptions: {len(df)}.
    Top affected roads: {', '.join(top_roads)}.
    Serious disruptions: {severe_count}.
    """

    prompt = f"You are an AI traffic analyst. Analyze this TfL data and write a clear 3-sentence summary:\n{context}"
    response = llm.predict(prompt)

    with open("latest_summary.txt", "w") as f:
        f.write(response.strip())

    return response.strip()

# ----------------------------
# 💬 Tool 3: Semantic Q&A
# ----------------------------
@tool("respond_to_user")
def respond_to_user(query: str) -> str:
    """Answer user questions about live traffic using semantic search."""
    conn = sqlite3.connect("cityflow.db")
    df = pd.read_sql("SELECT * FROM road_status", conn)
    conn.close()

    if df.empty:
        return "No TfL data available at the moment."

    df["text"] = df.apply(lambda r: f"{r['road']} — {r['severity']} — {r['category']} — {r['comments']}", axis=1)
    q_vec = embedder.encode([query])
    sims = cosine_similarity(q_vec, embedder.encode(df["text"].tolist()))[0]
    top_contexts = [df.iloc[i]["text"] for i in np.argsort(sims)[-5:][::-1]]
    context = "\n".join(top_contexts)

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer concisely and factually."
    return llm.predict(prompt)

# ----------------------------
# 🧭 Tool 4: Recommend alternate routes
# ----------------------------
@tool("recommend_alternatives")
def recommend_alternatives(location: str) -> str:
    """Suggest alternate routes near a location (experimental)."""
    prompt = f"Suggest alternate routes around {location} in London using general traffic patterns."
    return llm.predict(prompt)

# ----------------------------
# 📈 Tool 5: Compare today vs yesterday
# ----------------------------
@tool("compare_trends")
def compare_trends() -> str:
    """Compare current day’s disruptions with the previous day."""
    conn = sqlite3.connect("cityflow.db")
    df = pd.read_sql("SELECT * FROM road_status", conn)
    conn.close()

    if df.empty:
        return "No TfL data available to compare."

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    if len(df["date"].unique()) < 2:
        return "Not enough historical data to compare days yet."

    latest_date = df["date"].max()
    prev_date = sorted(df["date"].unique())[-2]

    today = df[df["date"] == latest_date]
    yesterday = df[df["date"] == prev_date]

    diff = len(today) - len(yesterday)
    trend = "increased" if diff > 0 else "decreased"
    pct = abs(diff) / len(yesterday) * 100 if len(yesterday) > 0 else 0

    return f"Traffic disruptions have {trend} by {pct:.1f}% compared to yesterday ({len(today)} vs {len(yesterday)} total)."

# ----------------------------
# ⚠️ Tool 6: Analyze disruption causes
# ----------------------------
@tool("analyze_causes")
def analyze_causes() -> str:
    """Identify common disruption causes (e.g., roadworks, collisions)."""
    conn = sqlite3.connect("cityflow.db")
    df = pd.read_sql("SELECT * FROM road_status", conn)
    conn.close()

    if df.empty:
        return "No data available."

    cause_counts = df["category"].value_counts().head(3).to_dict()
    return f"Top disruption causes: {', '.join([f'{k} ({v})' for k, v in cause_counts.items()])}."

# ----------------------------
# 🔮 Tool 7: Forecast near-future traffic
# ----------------------------
@tool("forecast_trends")
def forecast_trends() -> str:
    """Predict short-term congestion trends based on recent data."""
    conn = sqlite3.connect("cityflow.db")
    df = pd.read_sql("SELECT * FROM road_status", conn)
    conn.close()

    if df.empty:
        return "No live data available to forecast."

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    counts = df.groupby("hour").size()

    if len(counts) < 6:
        return "Not enough hourly data for forecasting."

    last_3h_avg = counts.tail(3).mean()
    prev_3h_avg = counts.iloc[-6:-3].mean()
    change = last_3h_avg - prev_3h_avg
    trend = "increasing" if change > 0 else "decreasing"

    return f"Traffic disruptions are {trend} over the last 3 hours, suggesting {('worsening' if change > 0 else 'improving')} conditions this evening."

# ----------------------------
# ⚙️ Initialize Multi-Tool Agent
# ----------------------------
tools = [
    fetch_tfl_data,
    summarize_trends,
    respond_to_user,
    recommend_alternatives,
    compare_trends,
    analyze_causes,
    forecast_trends,
]

prompt = hub.pull("hwchase17/react")
react_agent = create_react_agent(llm, tools, prompt)
cityflow_agent = AgentExecutor(agent=react_agent, tools=tools, verbose=True)
