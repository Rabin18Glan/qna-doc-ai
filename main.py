import os
import getpass
import streamlit as st
from chat_models.chat_models import model
from semantic_search import semantic_search
from classification.tagging_prompt import tagging_prompt
from classification.structured_model import structured_llm
from chain.chain import Chain

# Load .env if exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Setup environment variables
os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key (optional): ")
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass("Enter your LangSmith Project Name (default = 'default'): ") or "default"
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
if not os.environ.get("USER_AGENT"):
    os.environ["USER_AGENT"] = getpass.getpass("User Agent: ")

# Sidebar
with st.sidebar:
    st.title("📁 File & URL Manager")
    st.markdown("Upload your documents or add relevant URLs for chat context.")
    uploaded_files = st.file_uploader("📄 Upload Documents", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    urls_input = st.text_area("🌐 Add URLs (one per line)", height=120)

# Main Interface
st.markdown("<h1 style='text-align: center;'>💬 Chat with Your Files & Links</h1>", unsafe_allow_html=True)


# Input box
user_input = st.text_input("🔍 Ask a question:")

if user_input:
    url_list = [url.strip() for url in urls_input.splitlines() if url.strip()]

    # Create Chain
    chain = Chain(
        search=lambda q: semantic_search.run(q, uploaded_files=uploaded_files, urls=url_list),
        tagger=tagging_prompt,
        structured_model=structured_llm,
        chat_model=model
    )


    status_box = st.empty()

            # Process flow with real-time status updates
    for step in chain.process(user_input):
        status_box.caption(step)

    status_box.empty()
    st.write_stream(chain.invoke)
