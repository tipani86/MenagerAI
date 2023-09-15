import os
import asyncio
import streamlit as st
from inference import *
from pathlib import Path

FILE_ROOT = Path(__file__).parent

SUPPORTED_MODELS = {
    "gpt-3.5-turbo": {  # If you deployed on Azure, set the environment variables as below, and just ignore it if you deployed on OpenAI
        "model_name": os.getenv("AZURE_CHATGPT_MODEL_NAME", "gpt-3.5-turbo"),
        "model_engine": os.getenv("AZURE_CHATGPT_DEPLOYMENT_NAME", None),
        "model_func": call_openai,
        "avatar": "https://openai.com/favicon.ico",
        "stream": True
    },
    "llama-2-7b": {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "model_func": call_deepinfra,
        "avatar": "https://deepinfra.com/deepinfra-logo-64.webp",
        "stream": True
    },
    "llama-2-13b": {
        "model_name": "meta-llama/Llama-2-13b-chat-hf",
        "model_func": call_deepinfra,
        "avatar": "https://deepinfra.com/deepinfra-logo-64.webp",
        "stream": False
    },
    "llama-2-70b": {
        "model_name": "meta-llama/Llama-2-70b-chat-hf",
        "model_func": call_deepinfra,
        "avatar": "https://deepinfra.com/deepinfra-logo-64.webp",
        "stream": True
    },
    # "dolly-v2-12b": {
    #     "model_name": "databricks/dolly-v2-12b",
    #     "model_func": call_deepinfra,
    #     "avatar": "https://deepinfra.com/deepinfra-logo-64.webp",
    #     "stream": False
    # },
    # "falcon-7b": {"stream": False},
    # "pythia-12b": {"stream": False},
    # "pythia-2.8b": {"stream": False},
}

@st.cache_data(show_spinner=False)
def get_css() -> str:
    # Read CSS code from style.css file
    with open(FILE_ROOT / "style.css", "r") as f:
        return f"<style>{f.read()}</style>"
    
def render_messages(messages: list, assistant_avatar: str | None = None):
    for message in messages:
        match message["role"]:
            case "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            case "assistant":
                if assistant_avatar is None:
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("assistant", avatar=assistant_avatar):
                        st.markdown(message["content"])

async def get_reply(
    container: st.container,
    messages_key: str,
    prompt: str,
    model_settings: dict,
):
    with container:
        stream = model_settings.get("stream", False)
        
        # Render new human prompt
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add prompt to chat history
        st.session_state.messages[messages_key].append({"role": "user", "content": prompt})

        # Render new response
        with st.chat_message("assistant", avatar=model_settings.get("avatar", None)):
            message_placeholder = st.empty()
            full_response = ""
            message_placeholder.markdown(full_response + "▌")
            async for resp in model_settings["model_func"](
                messages=st.session_state.messages[messages_key],
                model_settings=model_settings,
            ):
                if stream:
                    if resp is not None:
                        full_response += resp
                        message_placeholder.markdown(full_response + "▌")
                else:
                    for chunk in resp.split(" "):
                        full_response += chunk + " "
                        message_placeholder.markdown(full_response + "▌")
                        await asyncio.sleep(0.01)
            message_placeholder.markdown(full_response)

        # Add response to chat history
        st.session_state.messages[messages_key].append({"role": "assistant", "content": full_response})

async def main():
    st.set_page_config(
        page_title="MenagerAI - Your Model Zoo",
        page_icon="🧊",
        layout="wide",
    )

    if "messages" not in st.session_state:
        st.session_state.messages = {}

    # Load CSS code
    st.markdown(get_css(), unsafe_allow_html=True)

    st.markdown("# Menager<i>AI</i>", unsafe_allow_html=True)

    st.write("Select up to four different models to compare side-by-side:")

    # Number of models to compare
    n_models = st.number_input("Number of models", 1, 4, 2, 1)

    # User-entered prompt for the models
    if n_models > 1:
        placeholder_text = f"Send a message to all {n_models} models"
    else:
        placeholder_text = "Send a message to the model"

    with st.form("prompt_form", clear_on_submit=True):
        prompt = st.text_area(
            "Prompt",
            placeholder=placeholder_text, 
            label_visibility="collapsed",
            height=50,
        )
        prompt_submitted = st.form_submit_button("Send")
    if st.button("Clear all message histories"):
        st.session_state.messages = {}

    # Render model columns
    columns = st.columns(n_models)

    for i, column in enumerate(columns):
        model = column.selectbox(f"Model {i+1}", sorted(SUPPORTED_MODELS.keys()), key=f"model_{i}")
        if f"{model}_{i}" not in st.session_state.messages:
            st.session_state.messages[f"{model}_{i}"] = []
        else:
            with column:
                render_messages(
                    st.session_state.messages[f"{model}_{i}"], 
                    SUPPORTED_MODELS[model].get("avatar", None)
                )

    if prompt_submitted:
        # Collect async tasks for all columns and run them simultaneously using gather
        coroutines = []
        for i in range(n_models):
            model = st.session_state[f"model_{i}"]
            messages_key = f"{model}_{i}"
            coroutines.append(get_reply(columns[i], messages_key, prompt, SUPPORTED_MODELS[model]))
        await asyncio.gather(*coroutines)

if __name__ == "__main__":
    asyncio.run(main())