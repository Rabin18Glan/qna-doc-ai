import os
import getpass
import streamlit as st
from chat_models.chat_models import model
from chat_models import chat_prompts
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

# Setup env vars
os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key (optional): ")
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass("Enter your LangSmith Project Name (default = 'default'): ") or "default"
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
# ... [imports and env setup as before] ...

steps = [
    "Semantic search (from provided documents or URLs)",
    "Tagging (Create custom tags)",
    "Structured classification",
    "Final Prompt"
]

# Sidebar
with st.sidebar:
    st.title("📂 Document & Link Manager")
    uploaded_files = st.file_uploader("📄 Upload files", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    urls_input = st.text_area("🌐 Add URLs (one per line)", height=120)
    # st.markdown("📝 Uploaded Files:")
    # if uploaded_files:
    #     for f in uploaded_files:
    #         st.markdown(f"• {f.name}")
    # st.markdown("🔗 Added URLs:")
    # for url in urls_input.strip().splitlines():
    #     if url.strip():
    #         st.markdown(f"• {url.strip()}")

# Main Chat Area
st.title("💬 Chat with Your Files, Websites & YouTube")
user_input = st.text_input("Ask your question:")

if user_input:
    url_list = [url.strip() for url in urls_input.splitlines() if url.strip()]

    chain = Chain(
        search=lambda q: semantic_search.run(q, uploaded_files=uploaded_files, urls=url_list),
        tagger=tagging_prompt,
        structured_model=structured_llm,
        chat_model=model
    )

    # Placeholders for live rendering
    step_placeholders = [st.empty() for _ in steps]
    final_placeholder = st.empty()
    full_response = ""
    intermediate_outputs = []

    step_index = 0
    stream_started = False

    for msg in chain.invoke(user_input):
        if step_index < len(steps):
            step_placeholders[step_index].subheader(f"🔹 {steps[step_index]}")
            step_placeholders[step_index].write(msg)
            intermediate_outputs.append((steps[step_index], msg))
            step_index += 1
        else:
            # First token from chat model — hide previous steps
            if not stream_started:
                for p in step_placeholders:
                    p.empty()
                final_placeholder.subheader("✅ Final Response")
                stream_started = True

            full_response += msg
            final_placeholder.write(full_response)

    # Optional expander to view earlier steps after completion
    with st.expander("🔍 Show Processing Steps"):
        for title, content in intermediate_outputs:
            st.subheader(f"🔹 {title}")
            st.write(content)
