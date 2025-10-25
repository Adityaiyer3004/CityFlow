"""
CITYFLOW AGENT — with FAISS vector memory
LangChain 0.2.16 / Core 0.2.38 / Smith 0.1.147
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
import os, sqlite3, threading, time
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field
import faiss
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import StructuredTool
from langchain.schema import AgentFinish

# ----------------------------------------------------------------------
# 🧠 Smart Model Loader (with preload + auto-fallback)
# ----------------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_smart_llm():
    """Initialize Groq LLM with fallback and retry."""
    global ACTIVE_MODEL
    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    current_index = 0

    def init_model(model_name):
        global ACTIVE_MODEL
        ACTIVE_MODEL = model_name
        print(f"🧠 Loading model: {model_name}")
        return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_name, temperature=0.4)

    llm = init_model(models[current_index])

    def safe_invoke(prompt):
        nonlocal llm, current_index
        for attempt in range(3):
            try:
                return llm.invoke(prompt)
            except Exception as e:
                msg = str(e).lower()
                if "rate limit" in msg or "429" in msg:
                    print(f"⚠️ Rate limit hit for {models[current_index]}")
                    current_index = (current_index + 1) % len(models)
                    llm = init_model(models[current_index])
                    continue
                elif "timeout" in msg or "503" in msg:
                    wait = 5 * (attempt + 1)
                    print(f"⏳ Timeout hit, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    raise
        raise RuntimeError("❌ All model attempts failed after retries.")

    def auto_refresh():
        nonlocal llm, current_index
        while True:
            time.sleep(600)
            try:
                if current_index != 0:
                    print("♻️ Restoring primary model...")
                    llm = init_model(models[0])
                    current_index = 0
                    print("✅ Restored primary model.")
            except Exception as e:
                print(f"⚠️ Auto-refresh failed: {e}")

    threading.Thread(target=auto_refresh, daemon=True).start()
    return llm, safe_invoke


class SafeLLM(BaseChatModel):
    """Wrapper with safe retry and fallback."""
    llm: Any = None
    safe_invoke: Any = None

    def __init__(self, llm, safe_invoke):
        super().__init__()
        object.__setattr__(self, "llm", llm)
        object.__setattr__(self, "safe_invoke", safe_invoke)
        print("✅ SafeLLM initialized successfully")

    def _generate(self, messages: list[BaseMessage], **kwargs):
        prompt = "\n".join([m.content for m in messages if m.content])
        resp = self.safe_invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        from langchain_core.outputs import ChatGeneration, ChatResult
        return ChatResult(generations=[ChatGeneration(message=self._to_chat_message(text))])

    def _to_chat_message(self, text):
        from langchain_core.messages import AIMessage
        return AIMessage(content=text)

    @property
    def _llm_type(self) -> str:
        return "safe_groq_llm"


llm, safe_invoke = get_smart_llm()
london_tz = pytz.timezone("Europe/London")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------------------------------------------------
#  2. MEMORY SETUP (FAISS + SQLite)
# ----------------------------------------------------------------------
FAISS_PATH = "cityflow_memory.index"
MEMORY_DB = "cityflow_memory.db"
embedding_dim = 384

if os.path.exists(FAISS_PATH):
    index = faiss.read_index(FAISS_PATH)
else:
    index = faiss.IndexFlatL2(embedding_dim)

conn = sqlite3.connect(MEMORY_DB)
conn.execute("""
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()
conn.close()


def add_to_memory(text: str):
    vec = embedder.encode([text]).astype("float32")
    index.add(vec)
    faiss.write_index(index, FAISS_PATH)
    conn = sqlite3.connect(MEMORY_DB)
    conn.execute("INSERT INTO memory (text) VALUES (?)", (text,))
    conn.commit()
    conn.close()


def retrieve_memory(query: str, top_k=3) -> list[str]:
    conn = sqlite3.connect(MEMORY_DB)
    df = pd.read_sql("SELECT * FROM memory", conn)
    conn.close()
    if df.empty:
        return []
    q_vec = embedder.encode([query]).astype("float32")
    mem_vecs = embedder.encode(df["text"].tolist()).astype("float32")
    index_temp = faiss.IndexFlatL2(mem_vecs.shape[1])
    index_temp.add(mem_vecs)
    D, I = index_temp.search(q_vec, top_k)
    return [df.iloc[i]["text"] for i in I[0] if i < len(df)]

# ----------------------------------------------------------------------
#  3. TOOL INPUT SCHEMAS + UTILS
# ----------------------------------------------------------------------
class QueryInput(BaseModel):
    query: str = Field(...)


class LocationInput(BaseModel):
    location: str = Field(...)


class EmptyInput(BaseModel):
    dummy: str = Field(default="")


def get_time_context():
    now = datetime.now(london_tz)
    hour = now.hour
    greeting = "Good morning" if 5 <= hour < 12 else "Good afternoon" if 12 <= hour < 18 else "Good evening"
    return greeting, now.strftime("%A, %H:%M %p")

# ----------------------------------------------------------------------
#  4. TOOLS
# ----------------------------------------------------------------------
def fetch_tfl_data(dummy=""):
    """Fetch the latest TfL road disruption data from the local SQLite database."""
    conn = sqlite3.connect("cityflow.db")
    df = pd.read_sql("SELECT * FROM road_status", conn)
    conn.close()
    if df.empty:
        return "No data found."
    latest = df["timestamp"].max()
    return f"Fetched {len(df)} disruptions. Latest update: {latest}"


def summarize_trends(dummy=""):
    """Summarize current London road disruptions using the TfL data."""
    conn = sqlite3.connect("cityflow.db")
    df = pd.read_sql("SELECT * FROM road_status", conn)
    conn.close()

    if df.empty:
        text = "No recent TfL data."
    else:
        top_roads = df["road"].value_counts().head(3).index.tolist()
        severe = (df["severity"] == "Serious").sum()
        context = f"Total disruptions: {len(df)}, Top roads: {', '.join(top_roads)}, Serious: {severe}"
        greeting, time_str = get_time_context()
        prompt = f"{greeting}, London. It is {time_str}. Summarize briefly: {context}"
        resp = safe_invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)

    add_to_memory(text)
    return text.strip()


def respond_to_user(query: str):
    """Answer user questions about current traffic conditions using TfL data and memory context."""
    past = retrieve_memory(query)
    memory_context = "\n".join(past) if past else "No relevant past context."

    conn = sqlite3.connect("cityflow.db")
    df = pd.read_sql("SELECT * FROM road_status", conn)
    conn.close()
    if df.empty:
        return "No TfL data available."

    df["text"] = df.apply(lambda r: f"{r['road']} — {r['severity']} — {r['category']} — {r['comments']}", axis=1)
    q_vec = embedder.encode([query])
    sims = cosine_similarity(q_vec, embedder.encode(df["text"].tolist()))[0]
    context = "\n".join([df.iloc[i]["text"] for i in np.argsort(sims)[-5:][::-1]])

    greeting, time_str = get_time_context()
    prompt = f"{greeting}, it's {time_str}. Context:\n{context}\nQuestion: {query}\nAnswer clearly."
    resp = safe_invoke(prompt)
    answer = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
    add_to_memory(answer)
    return answer


def recommend_alternatives(location: str):
    """Suggest alternate routes or nearby detours for a given location."""
    greeting, time_str = get_time_context()
    prompt = f"{greeting}, it's {time_str}. Suggest alternate routes near {location}."
    resp = safe_invoke(prompt)
    text = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
    add_to_memory(text)
    return text

# ----------------------------------------------------------------------
#  5. AGENT (LangGraph-based reasoning flow)
# ----------------------------------------------------------------------
from langgraph.graph import StateGraph
from typing import TypedDict

# Define state container
class AgentState(TypedDict):
    user_query: str
    tfl_summary: str
    trend_summary: str
    final_answer: str


# Node 1: Fetch TfL data
def node_fetch_tfl(state: AgentState) -> AgentState:
    state["tfl_summary"] = fetch_tfl_data()
    print("📊 TfL data fetched.")
    return state


# Node 2: Summarize trends
def node_summarize_trends(state: AgentState) -> AgentState:
    state["trend_summary"] = summarize_trends()
    print("🧾 Trend summary ready.")
    return state


# Node 3: Generate final answer
def node_generate_answer(state: AgentState) -> AgentState:
    query = state["user_query"]
    greeting, time_str = get_time_context()
    prompt = f"""
{greeting}, it's {time_str} in London.

TfL summary:
{state['tfl_summary']}

Trend summary:
{state['trend_summary']}

User question: {query}

Respond clearly and concisely in 2–3 sentences.
"""
    resp = safe_invoke(prompt)
    ans = getattr(resp, "content", str(resp)).strip()
    for marker in ["Thought:", "Action:", "Observation:", "Final Answer:"]:
        ans = ans.replace(marker, "")
    ans = " ".join(ans.split())
    state["final_answer"] = ans or "⚠️ No clear response generated."
    print(f"✅ Generated clean answer: {state['final_answer'][:200]}...")
    add_to_memory(state["final_answer"])
    return state


# -------------------------------
# Build LangGraph (v0.1 syntax)
# -------------------------------
graph = StateGraph(AgentState)
graph.add_node("FetchTfL", node_fetch_tfl)
graph.add_node("SummarizeTrends", node_summarize_trends)
graph.add_node("GenerateAnswer", node_generate_answer)

graph.add_edge("FetchTfL", "SummarizeTrends")
graph.add_edge("SummarizeTrends", "GenerateAnswer")
graph.set_entry_point("FetchTfL")

# Compile (old API)
app = graph.compile()


# -------------------------------
# Runner function
# -------------------------------
def run_cityflow_agent(inputs: dict):
    """Execute the LangGraph pipeline once for a given query."""
    try:
        query = inputs.get("input", "") or inputs.get("query", "")
        state: AgentState = {
            "user_query": query,
            "tfl_summary": "",
            "trend_summary": "",
            "final_answer": "",
        }

        # --- Run the agent ---
        final_state = app.invoke(state)

        # --- Determine which tool was most recently executed ---
        if final_state.get("trend_summary"):
            tool_used = "Summarize Trends"
        elif final_state.get("tfl_summary"):
            tool_used = "Fetch TfL Data"
        else:
            tool_used = "LLM Query"

        return {
            "output": final_state["final_answer"],
            "tool_used": tool_used,
            "intermediate_steps": [
                ("Fetched TfL data", final_state.get("tfl_summary", "")),
                ("Summarized trends", final_state.get("trend_summary", "")),
            ],
        }

    except Exception as e:
        err = f"⚠️ CityFlowAgent error: {e}"
        print(err)
        return {"output": err, "tool_used": "Error"}


# ✅ Set global reference for API
cityflow_agent = type("CityFlowAgent", (), {"invoke": run_cityflow_agent})()
