import os
import requests
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv
from datetime import datetime
import pytz
from textwrap import shorten
import sqlite3

# -----------------------------
# ⚙️ 1. Load environment variables
# -----------------------------
load_dotenv()
APP_KEY = os.getenv("TFL_APP_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not APP_KEY:
    print("⚠️ Warning: Missing TfL API key in .env")
if not GROQ_API_KEY:
    print("⚠️ Warning: Missing Groq API key in .env")

# Database setup
engine = create_engine("sqlite:///cityflow.db")
DB_PATH = "cityflow.db"


# -----------------------------
# 🧱 2. Initialize SQLite Schema
# -----------------------------
def init_db():
    """Ensure road_status table exists and matches the expected schema."""
    inspector = inspect(engine)
    table_name = "road_status"
    expected_cols = [
        "timestamp", "road", "severity", "category",
        "comments", "start_date", "end_date"
    ]

    if table_name in inspector.get_table_names():
        existing_cols = [c["name"] for c in inspector.get_columns(table_name)]
        if set(expected_cols) - set(existing_cols):
            print("⚠️ Schema mismatch — recreating table.")
            with engine.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))

    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                road TEXT,
                severity TEXT,
                category TEXT,
                comments TEXT,
                start_date TEXT,
                end_date TEXT
            );
        """))
    print("✅ road_status table ready.")


# -----------------------------
# 🚦 3. Fetch TfL Road Data
# -----------------------------
def fetch_tfl_data():
    """Fetch live TfL road disruption data and save to SQLite."""
    if not APP_KEY:
        raise ValueError("❌ Missing TfL API key")

    url = f"https://api.tfl.gov.uk/Road/all/Disruption?app_key={APP_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"TfL API error: {response.status_code}")

    data = response.json()
    if not data:
        print("⚠️ No disruption data received from TfL.")
        return pd.DataFrame()

    london_tz = pytz.timezone("Europe/London")
    now_uk = datetime.now(london_tz).strftime("%Y-%m-%d %H:%M:%S")

    records = []
    for road in data:
        records.append({
            "timestamp": now_uk,
            "road": road.get("location"),
            "severity": road.get("severity"),
            "category": road.get("category"),
            "comments": road.get("comments"),
            "start_date": road.get("startDateTime"),
            "end_date": road.get("endDateTime"),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df.to_sql("road_status", con=engine, if_exists="append", index=False)
        print(f"✅ Saved {len(df)} TfL records to SQLite at {now_uk}")
    else:
        print("⚠️ No valid records to save.")
    return df


# -----------------------------
# 🧠 4. Generate AI Summary
# -----------------------------
def generate_ai_summary(df: pd.DataFrame):
    """Generate a BBC-style London traffic report using Groq’s LLM."""
    if df.empty:
        summary = "⚠️ No data available to summarize."
        print(summary)
        return summary

    # --- London time awareness ---
    london_tz = pytz.timezone("Europe/London")
    now_london = datetime.now(london_tz)
    hour = now_london.hour

    # Greeting
    if hour < 12:
        greeting = "Good morning"
    elif hour < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    # --- TfL data snippet ---
    summary_text = "\n".join(
        f"{row.road or 'Unknown'} — {row.severity or 'N/A'} — {row.category or 'N/A'} — {row.comments or ''}"
        for _, row in df.head(15).iterrows()
    )

    # --- Dynamic tone selection ---
    if 6 <= hour < 12:
        tone_label = "Formal Morning Bulletin"
        tone_instruction = """
        You are a BBC London traffic reporter providing a morning rush-hour update.
        Use a crisp, informative tone with a sense of urgency but remain calm and clear.
        Focus on accuracy, brevity, and the key commuter routes causing the most disruption.
        """
    elif 12 <= hour < 17:
        tone_label = "Neutral Midday Update"
        tone_instruction = """
        You are a BBC London traffic reporter giving a midday travel update.
        Use a neutral, professional tone — focus on summarising ongoing works,
        moderate congestion, and give context for lunchtime or mid-day travellers.
        """
    else:
        tone_label = "Conversational Evening Radio Update"
        tone_instruction = """
        You are a BBC Radio London presenter giving a live evening travel update.
        Use a warm, conversational tone — sound like you're on-air during drive-time.
        Be friendly but informative, mention major delays, and end with helpful commuter advice.
        """

    # --- Unified prompt using tone_instruction ---
    prompt = f"""
    {tone_instruction}

    Current time: {now_london.strftime('%H:%M')} (London time)
    Greeting: "{greeting}, London."
    
    Data sample:
    {summary_text}

    Guidance:
    - Keep it 3–5 sentences.
    - Mention major congestion, closures, and ongoing works.
    - Avoid robotic phrasing or repetition.
    - End naturally, with advice like “allow extra time” or “check TfL updates before you go”.
    """

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "You are a BBC London traffic reporter generating live updates in British English."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 300,
                "temperature": 0.6,
            },
            timeout=20,
        )

        if response.status_code == 200:
            summary = response.json()["choices"][0]["message"]["content"].strip()
        else:
            print(f"⚠️ Groq API returned {response.status_code}")
            summary = shorten(" ".join(df["comments"].dropna()), width=400, placeholder="...")

    except Exception as e:
        summary = f"⚠️ Error generating summary: {e}"

    # --- Save to file ---
    with open("latest_summary.txt", "w") as f:
        f.write(
            f"🕒 Summary generated at {now_london.strftime('%Y-%m-%d %H:%M:%S')} (London time)\n"
        )
        f.write(summary)

    print(f"✅ latest_summary.txt updated ({tone_label}).")
    return summary

# -----------------------------
# 📈 5. Log Daily Disruptions
# -----------------------------
def log_daily_disruptions():
    """Logs total daily disruption count into daily_trends."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS daily_trends (
                date TEXT PRIMARY KEY,
                disruptions INTEGER
            )
        """)

        cur.execute("SELECT COUNT(*) FROM road_status")
        count = cur.fetchone()[0]

        today = datetime.now(pytz.timezone("Europe/London")).strftime("%Y-%m-%d")
        cur.execute("""
            INSERT OR REPLACE INTO daily_trends (date, disruptions)
            VALUES (?, ?)
        """, (today, count))
        conn.commit()
    print(f"📊 Logged {count} disruptions for {today} in daily_trends.")


# -----------------------------
# 🚀 6. Main Entry
# -----------------------------
if __name__ == "__main__":
    print("🚀 Running TfL data pipeline...")
    init_db()
    df = fetch_tfl_data()
    generate_ai_summary(df)
    log_daily_disruptions()
    print("✅ Pipeline completed successfully.")
