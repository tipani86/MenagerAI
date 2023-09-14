import os
import asyncio
import streamlit as st
from pathlib import Path

FILE_ROOT = Path(__file__).parent

SUPPORTED_MODELS = {
    "gpt-3.5-turbo": {  # If you deployed on Azure, set the environment variables as below, and just ignore it if you deployed on OpenAI
        "model_name": os.getenv("AZURE_CHATGPT_MODEL_NAME", "gpt-3.5-turbo"),
        "model_engine": os.getenv("AZURE_CHATGPT_DEPLOYMENT_NAME", None),
        "stream": True
    },
    "llama-2-7b": {"stream": True},
    "llama-2-13b": {"stream": True},
    "llama-2-70b": {"stream": True},
    "dolly-v2-12b": {"stream": False},
    "falcon-7b": {"stream": False},
    "pythia-12b": {"stream": False},
    "pythia-2.8b": {"stream": False},
}

@st.cache_data(show_spinner=False)
def get_css() -> str:
    # Read CSS code from style.css file
    with open(FILE_ROOT / "style.css", "r") as f:
        return f"<style>{f.read()}</style>"

async def _get_reply(container: st.container, messages: list, model_func: callable, model_func_kwargs: dict):
    with container:
        st.write(f"You wrote:")
        reply_box = st.empty()
        reply_text = ""
        for char in prompt:
            reply_text += char
            reply_box.write(reply_text)
            await asyncio.sleep(0.1)

async def main():
    st.set_page_config(
        page_title="MenagerAI",
        page_icon="🧊",
        layout="wide",
    )

    if "messages" not in st.session_state:
        st.session_state.messages = {}

    # Load CSS code
    st.markdown(get_css(), unsafe_allow_html=True)

    st.markdown("# Menager<i>AI</i>", unsafe_allow_html=True)

    st.write("Select up to four different models to compare side-by-side:")

    n_models = st.number_input("Number of models", 1, 4, 2, 1)

    columns = st.columns(n_models)

    for i, column in enumerate(columns):
        model = column.selectbox(f"Model {i+1}", sorted(SUPPORTED_MODELS.keys()), key=f"model_{i}")
        if f"{model}_{i}" not in st.session_state.messages:
            st.session_state.messages[f"{model}_{i}"] = []

    if n_models > 1:
        placeholder_text = f"Send a message to all {n_models} models"
    else:
        placeholder_text = "Send a message to the model"
    prompt = st.chat_input(placeholder_text)

    if prompt:
        # Collect async tasks for all columns and run them simultaneously using gather
        coroutines = []
        for i in range(n_models):
            model = st.session_state[f"model_{i}"]
            messages = st.session_state.messages[f"{model}_{i}"]
            messages.append({"role": "user", "content": prompt})
            coroutines.append(_get_reply(columns[i], messages, None))
        await asyncio.gather(*coroutines)

if __name__ == "__main__":
    asyncio.run(main())