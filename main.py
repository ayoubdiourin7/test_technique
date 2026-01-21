import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Cabinet Emilia Parenti — RAG PoC",
    page_icon="📚",
    layout="wide",
)

# Redirect straight to the chat page so only the two functional tabs appear.
try:
    st.switch_page("pages/1_Chat.py")
except Exception:
    # Fallback if switch_page is unavailable; offer manual navigation.
    st.write("Redirection vers la page Chat… utilisez le menu latéral si nécessaire.")
