import streamlit as st
from langchain_openai import ChatOpenAI

def get_llm(**kw):
    """
    ChatOpenAI LLM instance using Streamlit secrets.
    """
    return ChatOpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        model_name="gpt-4o",  # OpenaI Model name
        **kw,
    )