import sys, os, asyncio, json, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------
# 🧩 Local Imports
# ---------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tfl_pipeline import init_db
from app.cityflow_agent import cityflow_agent, summarize_trends
from app.routes import traffic

# ---------------------------------------------------------
# ⚙️ FastAPI App Setup
# ---------------------------------------------------------
app = FastAPI(
    title="CityFlow API",
    description="🚦 AI-powered London Traffic Insights Backend",
    version="1.0"
)

# ✅ Register Routers
app.include_router(traffic.router, prefix="/api")

# Thread executor for blocking LLM operations
executor = ThreadPoolExecutor(max_workers=2)

# ---------------------------------------------------------
# 🔄 Background Auto-Refresh
# ---------------------------------------------------------
async def auto_refresh_summary():
    """Periodically refresh TfL DB + AI summary every 15 minutes."""
    while True:
        try:
            print("♻️ Refreshing TfL data + AI summary...")
            init_db()
            summarize_trends()
            print("✅ Auto-refresh completed successfully.")
        except Exception as e:
            print(f"⚠️ Auto-refresh failed: {e}")
        await asyncio.sleep(900)  # 15 minutes


@app.on_event("startup")
def startup_event():
    print("🚀 Initializing CityFlow database...")

    try:
        # 1️⃣ Initialize DB and generate initial summary
        init_db()
        summarize_trends()
        print("🧠 Initial AI summary generated successfully.")
    except Exception as e:
        print(f"⚠️ Startup summary generation failed: {e}")

    try:
        # 2️⃣ Warm up LLM early so first query is instant
        print("🔥 Warming up Groq model...")
        safe_invoke("Warm-up: test prompt to load weights")
        print("✅ Model warm-loaded and ready.")
    except Exception as e:
        print(f"⚠️ Warm-up failed: {e}")

    try:
        # 3️⃣ Launch background refresher (once)
        loop = asyncio.get_event_loop()
        loop.create_task(auto_refresh_summary())
        print("✅ Database initialized & background updater running.")
    except Exception as e:
        print(f"⚠️ Could not start auto-refresh: {e}")


# ---------------------------------------------------------
# 🌍 CORS
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# 📍 API ENDPOINTS
# ---------------------------------------------------------
@app.get("/api/health")
def health_check():
    """Simple health check endpoint."""
    db_exists = os.path.exists("cityflow.db")
    return {"status": "ok" if db_exists else "initializing"}


@app.get("/api/summary")
def get_summary():
    """Return the latest AI-generated traffic summary."""
    try:
        with open("latest_summary.txt", "r") as f:
            summary = f.read().strip()
        return {"summary": summary, "status": "success"}
    except FileNotFoundError:
        return {"summary": "⚠️ No AI summary available yet.", "status": "pending"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@app.post("/api/ask")
async def ask_agent(request: Request):
    """
    Main endpoint to handle agent queries.
    Accepts JSON: { "input": "<query>" }
    """
    try:
        raw = await request.body()
        print("📦 RAW BODY:", raw)

        try:
            data = json.loads(raw.decode())
        except Exception:
            parsed = urllib.parse.parse_qs(raw.decode())
            data = {k: v[0] for k, v in parsed.items()}

        user_input = data.get("input")
        if not user_input:
            raise HTTPException(status_code=400, detail="Missing 'input' field in request.")

        print(f"🧠 Received Query: {user_input}")

        # ---------------------------------------------------------
        # ✅ Run agent safely (returns full dict)
        # ---------------------------------------------------------
        def safe_invoke_agent(query: str):
            try:
                result = cityflow_agent.invoke({"input": query})
                if isinstance(result, dict):
                    return result
                else:
                    return {"output": str(result), "tool_used": "LLM Query", "intermediate_steps": []}
            except Exception as e:
                print(f"⚠️ [SafeInvoke Error] {e}")
                return {"output": f"⚠️ Error while invoking agent: {e}", "tool_used": "Error", "intermediate_steps": []}

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, lambda: safe_invoke_agent(user_input))

        # ---------------------------------------------------------
        # ✅ Extract all relevant fields dynamically
        # ---------------------------------------------------------
        answer = result.get("output", "⚠️ No output.")
        tool_used = result.get("tool_used", "LLM Query")
        intermediate_steps = result.get("intermediate_steps", [])

        print(f"✅ Final Answer: {answer[:150]}... | Tool: {tool_used}")

        return {
            "query": user_input,
            "answer": answer.strip(),
            "output": answer.strip(),
            "tool_used": tool_used,
            "intermediate_steps": intermediate_steps,
            "status": "success",
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"❌ Error during /api/ask: {e}")
        return {"error": str(e), "status": "failed"}
