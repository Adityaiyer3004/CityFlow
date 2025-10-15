# 🚦 CityFlow – AI-Powered London Traffic Dashboard

CityFlow is an **AI-driven Streamlit dashboard** that visualizes real-time **TfL (Transport for London)** disruptions and allows users to **chat with an intelligent traffic assistant** powered by **LLMs and embeddings**.

---

## 🌍 Overview

CityFlow connects to the TfL API, stores disruption data in SQLite, and updates every 15 minutes.  
It then uses **sentence-transformer embeddings** to enable **natural-language queries** like:

> “What’s happening near Westminster and Southwark?”  
> “Show me all road closures today.”

This makes it the perfect fusion of **data visualization**, **live monitoring**, and **AI conversation**.

---

## 🧩 Features

- 🔄 **Auto-refreshing data** from TfL every 15 minutes  
- 🧠 **AI-generated summaries** of live London traffic conditions  
- 💬 **CityFlow Assistant** – a smart chatbot using `sentence-transformers` + `Groq Llama 3.3`  
- 📈 **Interactive charts** built with Plotly (severity breakdowns, time trends)  
- ⚙️ **SQLite backend** for storing and refreshing TfL data  
- 🕐 **Timezone-aware updates** (London local time)  
- 🌗 **Modern, responsive Streamlit UI**

---

## 🧠 Tech Stack

| Layer | Tools |
|:------|:------|
| **Frontend** | Streamlit, Plotly |
| **Backend** | Python, SQLite |
| **AI/ML** | Sentence Transformers (`all-MiniLM-L6-v2`), Groq API (`Llama 3.3 70B`) |
| **Infra** | Python-dotenv, Requests, Pytz |
| **Embeddings** | Cosine Similarity for semantic retrieval |

---

## 🧰 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Adityaiyer3004/CityFlow.git
cd CityFlow


2️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Add your environment variables
Create a .env file in the root directory:

bash
Copy code
GROQ_API_KEY=your_groq_api_key_here
4️⃣ Run the dashboard
bash
Copy code
streamlit run cityflow_dashboard.py
💬 Example Queries for the Chatbot
“What’s the traffic like in Camden right now?”

“Are there any major disruptions in Westminster?”

“Show me all delays across Southwark.”

“What roads are closed near Canary Wharf?”

🚀 Project Structure
bash
Copy code
CityFlow/
│
├── cityflow_dashboard.py     # Streamlit dashboard (AI + charts)
├── tfl_pipeline.py           # TfL API data ingestion pipeline
├── cityflow.db               # SQLite database
├── latest_summary.txt        # AI-generated summary file
├── requirements.txt          # Project dependencies
├── data/                     # Data storage folder
│   └── london_boroughs.geojson (optional)
└── .env                      # API keys (not committed)

🧑‍💻 Author
Aditya Iyer
Data Scientist & AI Engineer
📍 London, UK
💼 LinkedIn : https://linkedin.com/in/aditya-iyer
🌐 GitHub : https://github.com/Adityaiyer3004

🛡️ Disclaimer
CityFlow uses public TfL data under the Open Government Licence (OGL).
This project is for educational and research purposes only.

⭐ If you like this project, give it a star!
