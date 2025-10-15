import os
import requests
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from datetime import datetime, timezone
import pytz
from textwrap import shorten

# -----------------------------
# 🚦 1. Load environment + keys
# -----------------------------
load_dotenv()
APP_KEY = os.getenv("TFL_APP_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("TfL App Key:", APP_KEY)
print("Groq Key Loaded:", bool(GROQ_API_KEY))

if not APP_KEY:
    raise ValueError("❌ Missing TfL API key in .env")
if not GROQ_API_KEY:
    raise ValueError("❌ Missing Groq API key in .env")

# Set timezone to London
london_tz = pytz.timezone("Europe/London")
now_uk = datetime.now(london_tz).strftime("%Y-%m-%d %H:%M:%S")    

# -----------------------------------------
# 🚗 2. Fetch live data from TfL disruptions
# -----------------------------------------
url = f"https://api.tfl.gov.uk/Road/all/Disruption?app_key={APP_KEY}"
response = requests.get(url)

print("Response code:", response.status_code)
print("Sample response snippet:", response.text[:250])

if response.status_code != 200:
    raise Exception(f"❌ API call failed: {response.status_code} - {response.text}")

data = response.json()

# Auto-detect and normalize to UTC
local_tz = datetime.now().astimezone().tzinfo  # detects your system timezone (e.g., Asia/Kolkata)
current_time_local = datetime.now(local_tz)
current_time_utc = current_time_local.astimezone(pytz.utc)

# Debugging info (optional)
print(f"📍 Local time ({local_tz}):", current_time_local.strftime("%Y-%m-%d %H:%M:%S"))
print("🌍 Converted UTC time:", current_time_utc.strftime("%Y-%m-%d %H:%M:%S"))


records = []
for road in data:
    records.append({
      "timestamp": current_time_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "road": road.get("location"),
        "severity": road.get("severity"),
        "category": road.get("category"),
        "comments": road.get("comments"),
        "start_date": road.get("startDateTime"),
        "end_date": road.get("endDateTime")
    })

df = pd.DataFrame(records)
print("\n🔹 Live TfL Disruption Data:")
print(df[["road", "severity", "category", "comments"]].head(10))

# Save to SQLite
engine = create_engine("sqlite:///cityflow.db")
df.to_sql("road_status", con=engine, if_exists="append", index=False)
print(f"\n✅ {len(df)} road records saved at {datetime.now(london_tz).strftime('%Y-%m-%d %H:%M:%S')} (London time)")

# -----------------------------------------
# 🧠 3. Groq AI Traffic Summary Generation
# -----------------------------------------
print("\n🧠 Generating AI Traffic Summary via Groq...")

summary_text = "\n".join(
    f"{row.road} — {row.severity} — {row.category} — {row.comments}"
    for _, row in df.head(15).iterrows()
)

prompt = f"""
You are a London traffic reporter.
Summarise these live TfL road disruptions clearly and concisely.
Mention which areas are most affected and note any severe works or closures.

{summary_text}
"""

url = "https://api.groq.com/openai/v1/chat/completions"

response = requests.post(
    url,
    headers={
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "llama-3.3-70b-versatile",  # ✅ Supported model
        "messages": [
            {"role": "system", "content": "You summarise live traffic updates concisely."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 250,
    },
)

print(f"\nGroq API Status: {response.status_code}")
print(f"Raw Response (first 400 chars):\n{response.text[:400]}")

# -----------------------------------------
# 🧩 4. Handle Success or Fallback
# -----------------------------------------
if response.status_code != 200:
    print("⚠️ Groq model failed, switching to fallback summariser...")
    fallback_summary = shorten(" ".join(df["comments"].dropna()), width=400, placeholder="...")
    print("\n🗞️ Fallback Traffic Summary:\n", fallback_summary)

    # 💾 Save fallback summary for dashboard
    with open("latest_summary.txt", "w") as f:
        f.write(f"🕒 Summary generated at {datetime.now(london_tz).strftime('%Y-%m-%d %H:%M:%S')} (London time)\n")
        f.write(fallback_summary)

else:
    try:
        summary = response.json()["choices"][0]["message"]["content"].strip()
        print("\n🗞️ CityFlow AI Traffic Summary (Groq Llama 3.3-70B):\n")
        print(summary)

        # 💾 Save summary for dashboard
        with open("latest_summary.txt", "w") as f:
            f.write(f"🕒 Summary generated at {datetime.now(london_tz).strftime('%Y-%m-%d %H:%M:%S')} (London time)\n")
            f.write(summary)

    except Exception as e:
        print("⚠️ Could not parse summary:", e)


    # -------------------------------
    # 🧠 Generate AI Summary (Simple Placeholder)
    # -------------------------------
    conn = sqlite3.connect("cityflow.db")
    df = pd.read_sql("SELECT * FROM road_status", conn)
    conn.close()

    summary = []
    summary_lines.append(f"🕒 Summary generated at {datetime.now(london_tz).strftime('%Y-%m-%d %H:%M:%S')} (London time)")

    if not df.empty:
        count_serious = (df['severity'] == 'Serious').sum()
        count_moderate = (df['severity'] == 'Moderate').sum()
        count_minimal = (df['severity'] == 'Minimal').sum()

        summary.append(
            f"🚗 There are currently {count_serious} serious, {count_moderate} moderate, "
            f"and {count_minimal} minimal disruptions across London."
        )
        summary.append("🧠 Overall, traffic conditions are stable with moderate congestion on key routes.")
    else:
        summary.append("⚠️ No data available from TfL at the moment.")

    with open("latest_summary.txt", "w") as f:
        f.write("\n".join(summary))

    print("✅ latest_summary.txt generated successfully (Fallback).")
