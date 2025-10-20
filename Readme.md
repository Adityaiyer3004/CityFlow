# 🚦 CityFlow – AI-Powered London Traffic Intelligence Agent

CityFlow uses **Groq Llama-3.3 70B**, **LangChain**, and **TfL Open Data** to create a real-time AI agent that analyses live London traffic.

---

## ⚙️ Features
- 🧠 **Agentic AI System** – Groq-powered reasoning across multiple tools  
- 🛰 **Live TfL Feed Integration** – refreshed every 15 min  
- 📊 **Interactive Dashboard** – Streamlit + Plotly visualizations  
- 💬 **Semantic Q&A + Trend Forecasts**  
- 🧩 **Agent Console** – Real-time insight into tool usage & latency  

---

## 🧰 Tech Stack
- **LangChain Community**
- **Groq API (Llama 3.3 70B)**
- **Streamlit + Plotly**
- **Sentence Transformers**
- **SQLite + Pandas**
- **Python 3.11**

cityflow/
├── app/
│   ├── __init__.py
│   ├── main.py                ← FastAPI entrypoint
│   ├── routes/
│   │   ├── __init__.py
│   │   └── traffic.py          ← all endpoints
│   ├── core/
│   │   ├── agent_logic.py      ← uses cityflow_agent + summarize_trends
│   │   └── tfl_utils.py        ← handles TfL data reading
│   └── db/
│       └── database.py         ← SQLite connection
├── cityflow_agent.py
├── cityflow_dashboard.py
├── requirements.txt
└── .env


## 🔮 Future Work

CityFlow is designed as a foundation for **next-generation urban intelligence systems**.  
Here are some exciting directions for future development:

### 🧩 1. Autonomous Mode (Self-RAG)
Let the agent re-analyze and summarize trends automatically every 15 minutes, no user input needed.

### 🌆 2. Real-Time Route Optimisation
Integrate with **Google Maps API** or **OpenRouteService** to recommend alternate routes dynamically based on congestion levels.

### 🧠 3. Multi-Agent Collaboration
Extend CityFlow into a **multi-agent system**, e.g., one agent summarising trends, another forecasting, and another verifying data quality.

### 🛰 4. Historical Pattern Learning
Incorporate a **Time-Series model (Prophet / LSTM)** trained on TfL data for predictive congestion forecasting.

### 🧾 5. Web Deployment + CI/CD
Deploy to **Streamlit Cloud** or **Render**, automate data refresh via **GitHub Actions**, and integrate observability with **LangSmith**.

### 🔗 6. Cross-City Expansion
Generalize the pipeline to include **Paris, Madrid, or Mumbai** using open government data feeds, turning CityFlow into a **global AI mobility monitor**.

## 👨‍💻 Author
**Aditya Iyer – Data Scientist & AI Engineer**
**Built with ❤️ using Groq + LangChain + TfL Open Data**
---

## 🚀 Run Locally
```bash
git clone https://github.com/adityaiyer30/CityFlow.git
cd CityFlow
pip install -r requirements.txt
python tfl_pipeline.py
streamlit run cityflow_dashboard.py
🧠 Example Queries
Query	Tool Used
“Summarise today’s traffic trends.”	summarize_trends
“Compare traffic today vs yesterday.”	compare_trends
“What’s causing most disruptions?”	analyze_causes
“Forecast congestion for this evening.”	forecast_trends
“Suggest alternate routes near Camden.”	recommend_alternatives



