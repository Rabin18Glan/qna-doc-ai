import time
from langchain.chat_models import init_chat_model
import streamlit as st
from google.api_core.exceptions import ServiceUnavailable


model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

