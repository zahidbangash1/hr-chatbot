import streamlit as st
import os
import sys
import tempfile

# Import your pipeline
from pipeline import HRPipelineOrchestrator
from exception import CustomException
from logger import logging

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="HR Assistant Chatbot",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------
# Title
# -------------------------------
st.title("🤖 HR Assistant Chatbot")
st.write("Upload your HR policy PDF and ask questions")

# -------------------------------
# Session State
# -------------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# Sidebar (Initialization)
# -------------------------------
st.sidebar.header("⚙️ Settings")

# PDF File Uploader
uploaded_pdf = st.sidebar.file_uploader(
    "📄 Upload HR Policy PDF",
    type=["pdf"],
    help="Upload your HR policy document in PDF format"
)

model_name = st.sidebar.text_input(
    "LLM Model",
    value="qwen/qwen3-32b"
)

if st.sidebar.button("🚀 Initialize Pipeline"):
    if uploaded_pdf is None:
        st.sidebar.error("❌ Please upload a PDF file first.")
    else:
        try:
            with st.spinner("Initializing pipeline... Please wait ⏳"):
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_pdf.getvalue())
                    temp_pdf_path = tmp_file.name

                pipeline = HRPipelineOrchestrator(
                    pdf_path=temp_pdf_path,
                    llm_model=model_name
                )
                pipeline.initialize_pipeline()

                st.session_state.pipeline = pipeline
                st.session_state.initialized = True
                st.session_state.temp_pdf_path = temp_pdf_path

            st.sidebar.success("✅ Pipeline initialized successfully!")

        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")

# Cleanup temp file on session end (optional safeguard)
if st.session_state.get("initialized") and "temp_pdf_path" in st.session_state:
    if not os.path.exists(st.session_state.temp_pdf_path):
        st.session_state.initialized = False
        st.session_state.pipeline = None

# -------------------------------
# Chat UI
# -------------------------------
st.subheader("💬 Chat")

# Show chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Input box
user_input = st.chat_input("Ask your question...")

if user_input:
    if not st.session_state.initialized:
        st.warning("⚠️ Please initialize the pipeline first.")
    else:
        # Show user message
        st.chat_message("user").markdown(user_input)

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        try:
            with st.spinner("Thinking... 🤔"):
                response = st.session_state.pipeline.ask_question(user_input)

            # Show bot response
            st.chat_message("assistant").markdown(response)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })

        except Exception as e:
            st.error(f"❌ Error: {e}")