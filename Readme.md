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



