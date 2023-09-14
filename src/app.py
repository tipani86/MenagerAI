import openai
import streamlit as st
from pathlib import Path

FILE_ROOT = Path(__file__).parent

SUPPORTED_MODELS = {
    "gpt-3.5-turbo": {"stream": True},
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

st.set_page_config(
    page_title="MenagerAI",
    page_icon="🧊",
    layout="wide",
)

# Load CSS code
st.markdown(get_css(), unsafe_allow_html=True)

st.title("Menager _AI_")

st.write("Select up to four different models to try side-by-side:")

n_models = st.number_input("Number of models", 1, 4, 2, 1)

columns = st.columns(n_models)

for i, column in enumerate(columns):
    model = column.selectbox(f"Model {i+1}", sorted(SUPPORTED_MODELS.keys()), key=f"model_{i}")

if n_models > 1:
    placeholder_text = f"Send a message to all {n_models} models"
else:
    placeholder_text = "Send a message to the model"
human_input = st.chat_input(placeholder_text)