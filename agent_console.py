import time
import streamlit as st

class AgentConsole:
    def __init__(self):
        self.logs = []

    def start(self, tool_name: str):
        """Start timing and log tool name."""
        self.current_tool = tool_name
        self.start_time = time.time()
        st.markdown(
            f"<div style='background:#1e293b; color:#e2e8f0; padding:8px 12px; border-radius:8px; margin-top:8px;'>"
            f"🧩 <b>Running Tool:</b> <code>{tool_name}</code></div>",
            unsafe_allow_html=True
        )

    def end(self, status="✅ Success"):
        """End timing and show result summary."""
        elapsed = round(time.time() - self.start_time, 2)
        color = "#16a34a" if "Success" in status else "#dc2626"
        st.markdown(
            f"<div style='background:#0f172a; color:#e2e8f0; padding:8px 12px; border-radius:8px; margin-top:4px;'>"
            f"<b style='color:{color};'>{status}</b> • 🕒 {elapsed}s • <code>{self.current_tool}</code></div>",
            unsafe_allow_html=True
        )
        self.logs.append({"tool": self.current_tool, "time": elapsed, "status": status})
