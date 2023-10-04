import os
import base64
import random
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
    "codellama-34b": {
        "model_name": "codellama/CodeLlama-34b-Instruct-hf",
        "model_func": call_deepinfra,
        "avatar": "https://deepinfra.com/deepinfra-logo-64.webp",
        "stream": True
    },
    "llama-2-7b": {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "model_func": call_deepinfra,
        "avatar": "https://deepinfra.com/deepinfra-logo-64.webp",
        "stream": True
    },
    "llama-2-70b": {
        "model_name": "meta-llama/Llama-2-70b-chat-hf",
        "model_func": call_deepinfra,
        "avatar": "https://deepinfra.com/deepinfra-logo-64.webp",
        "stream": True
    },
    "llama-2-13b": {
        "model_name": "meta-llama/Llama-2-13b-chat-hf",
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
    # "falcon-7b": {
    #     "model_name": "tiiuae/falcon-7b-q51",
    #     "model_func": call_deepinfra,
    #     "avatar": "https://deepinfra.com/deepinfra-logo-64.webp",
    #     "stream": False
    # },
    # "pythia-12b": {
    #     "model_name": "EleutherAI/pythia-12b",
    #     "model_func": call_deepinfra,
    #     "avatar": "https://deepinfra.com/deepinfra-logo-64.webp",
    #     "stream": False
    # },
    # "pythia-2.8b": {
    #     "model_name": "EleutherAI/pythia-2.8b",
    #     "model_func": call_deepinfra,
    #     "avatar": "https://deepinfra.com/deepinfra-logo-64.webp",
    #     "stream": False
    # },
}

# Automatically scroll to bottom of page
js = f"""
<script>
    function scroll(dummy_var_to_force_repeat_execution){{
        const streamlitDoc = window.parent.document;
        // Find the last div element with data-testid="stChatMessageContent" and scroll to it
        const chatMessages = streamlitDoc.querySelectorAll('[data-testid="stChatMessageContent"]');
        const lastMessage = chatMessages[chatMessages.length - 1];
        lastMessage.scrollIntoView({{behavior: "smooth"}});
    }}
    scroll({random.randint(1, 10000)})
</script>
"""


@st.cache_data(show_spinner=False)
def get_css() -> str:
    # Read CSS code from style.css file
    with open(FILE_ROOT / "style.css", "r") as f:
        return f"<style>{f.read()}</style>"

@st.cache_data(show_spinner=False)
def get_local_img(file_path: Path) -> str:
    # Load a byte image and return its base64 encoded string
    return base64.b64encode(open(file_path, "rb").read()).decode("utf-8")

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
    loading_fp: Path = FILE_ROOT / "loading.gif",
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
            message_placeholder.markdown(f"<img src='data:image/gif;base64,{get_local_img(loading_fp)}' width=30 height=10>", unsafe_allow_html=True)
            full_response = ""
            try:
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
            except Exception as e:
                message_placeholder.empty()
                st.error(e)
                # Remove the previous human prompt from chat history
                st.session_state.messages[messages_key].pop()
                if st.button("I understand"):
                    st.rerun()
                st.stop()

        # Add response to chat history
        st.session_state.messages[messages_key].append({"role": "assistant", "content": full_response})

async def main():
    st.set_page_config(
        page_title="MenagerAI - Your Model Zoo",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load CSS code
    st.markdown(get_css(), unsafe_allow_html=True)
    title = "# Menager<i>AI</i> 🦙 🦅 🐑"
    with st.sidebar:
        st.header("Try for Free")
        st.caption("Chat with up to four models for maximum 5 conversational rounds each, on me.")
        free_trial = st.button("Wow, much generous, very thanks!", disabled="n_conversations" in st.session_state or "openai_key" in st.session_state or "deepinfra_key" in st.session_state)
        st.header("Enter API keys")
        st.caption("For unlimited use, enter your OpenAI and DeepInfra API keys below. Then it will run on your own accounts and you can use it as much as you want.")
        openai_key = st.text_input("OpenAI API key", type="password", help="Register your account and get your OpenAI API key at https://platform.openai.com")
        deepinfra_key = st.text_input("DeepInfra API key", type="password", help="Register your account and get your DeepInfra API key at https://deepinfra.com")
        st.caption("_**Author's Note:** While I can only claim that your credentials are not stored anywhere, for maximum security, you should generate new app-specific API keys on your accounts and use them here. This way, you can deactivate the keys after you don't plan to use the app anymore, and it won't affect any of your other keys/apps. You can check out the GitHub source for this app using below button:_")
        st.markdown('<a href="https://github.com/tipani86/MenagerAI"><img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/tipani86/MenagerAI?style=social"></a>', unsafe_allow_html=True)
        st.markdown('<small>Page views: <img src="https://www.cutercounter.com/hits.php?id=hexoxdck&nd=4&style=2" border="0" alt="visitor counter"></small>', unsafe_allow_html=True)

    # Check if the user has entered both keys OR they pressed the free trial button
    if openai_key:
        st.session_state["openai_key"] = openai_key
        st.session_state["OPENAI_API_BASE"] = "https://api.openai.com/v1"
        st.session_state["OPENAI_API_TYPE"] = "openai"
        if "gpt-3.5-turbo" in SUPPORTED_MODELS:
            SUPPORTED_MODELS["gpt-3.5-turbo"]["model_name"] = "gpt-3.5-turbo"
            SUPPORTED_MODELS["gpt-3.5-turbo"]["model_engine"] = None
    if deepinfra_key:
        if deepinfra_key == "":
            del st.session_state["deepinfra_key"]
            st.rerun()
        st.session_state["deepinfra_key"] = deepinfra_key
    if "openai_key" in st.session_state and "deepinfra_key" in st.session_state:
        if "n_conversations" in st.session_state:
            del st.session_state["n_conversations"]
    elif "n_conversations" in st.session_state:
        pass
    elif free_trial:
        if "n_conversations" not in st.session_state:
            st.session_state["n_conversations"] = 0
        st.rerun()
    else:
        st.markdown(title, unsafe_allow_html=True)
        st.write("Please enter your API keys or press the free trial button on the sidebar to continue.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = {}

    if "n_conversations" in st.session_state:
        title += f"<small>Lite ({5 - st.session_state['n_conversations']}/5 conversations left)</small>"
    st.markdown(title, unsafe_allow_html=True)
    st.markdown("Select up to four different models to compare them side-by-side:")

    # Reminder about DeepInfra's time-based models
    st.info("Please note that although DeepInfra.com supports several more open source models not listed here (such as Dolly, Falcon), they are inference-time based instead of token based, and recently inference times have been quite extensive with them. Therefore, we have excluded them from the comparisons for now, but will keep monitoring and can add them back if cost becomes reasonable.", icon="⚠️")

    # Number of models to compare
    n_models = st.number_input("Number of models", 1, 4, 2, 1)

    # User-entered prompt for the models
    if n_models > 1:
        placeholder_text = f"Send a message to all {n_models} models"
    else:
        placeholder_text = "Send a message to the model"

    # Disable the form if the free trial user has reached the maximum number of conversations
    if "n_conversations" in st.session_state and st.session_state["n_conversations"] >= 5:
        st.info("You have reached the maximum number of conversations for the free trial. Please enter your API keys on the sidebar to continue.")
    else:
        with st.form("prompt_form", clear_on_submit=True):
            prompt = st.text_area(
                "Prompt",
                placeholder=placeholder_text, 
                label_visibility="collapsed",
                height=50,
            )
            prompt_submitted = st.form_submit_button("Send", type="primary")
        if st.button("Clear all message histories"):
            st.session_state.messages = {}

    # Render model columns
    columns = st.columns(n_models)
    all_models = sorted(SUPPORTED_MODELS.keys())
    # Move ChatGPT to the top of the list
    if "gpt-3.5-turbo" in all_models:
        all_models.insert(0, all_models.pop(all_models.index("gpt-3.5-turbo")))

    for i, column in enumerate(columns):
        model = column.selectbox(f"Model {i+1}", all_models, index=i, key=f"model_{i}")
        messages_key = f"{model}_{i}"
        if messages_key not in st.session_state.messages:
            st.session_state.messages[messages_key] = []
        else:
            with column:
                render_messages(
                    st.session_state.messages[messages_key], 
                    SUPPORTED_MODELS[model].get("avatar", None)
                )

    st.markdown("""<small><a target="_self" href="#menagerai">Back to Top</a></small>""", unsafe_allow_html=True)

    if "n_conversations" in st.session_state and st.session_state["n_conversations"] >= 5:
        st.stop()

    if prompt_submitted:
        if "n_conversations" in st.session_state:
            st.session_state["n_conversations"] += 1
        # Collect async tasks for all columns and run them simultaneously using gather
        coroutines = []
        for i in range(n_models):
            model = st.session_state[f"model_{i}"]
            messages_key = f"{model}_{i}"
            coroutines.append(get_reply(columns[i], messages_key, prompt, SUPPORTED_MODELS[model]))
        st.components.v1.html(js, height=0)
        await asyncio.gather(*coroutines)
        st.rerun()

if __name__ == "__main__":
    asyncio.run(main())