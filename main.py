import streamlit as st
from utils import get_rain_info
from openai import inference_gpt
from huggingface import inference_huggingface

# App framework
st.title('🦜🔗 บรรทัดทอง Chat Bot')
st.sidebar.title("Settings")

model_list = [
    'google/mt5-small',
    'deepset/roberta-base-squad2', 
    'meta-llama/Meta-Llama-3-8B', 
    'microsoft/Phi-3-mini-4k-instruct', 
    'gpt-3.5-turbo-0125'
]
model_name = st.sidebar.radio("Select model: ", model_list)
prompt = st.text_input('Enter your question here') 

# Show to the screen if there's a prompt
if prompt:
    if model_name == "gpt-3.5-turbo-0125":
        response = inference_gpt(prompt)
    elif model_name in model_list[1:]:
        response, retrieved_docs = inference_huggingface(prompt, model_name)
    else:
        response = ""
        retrieved_docs = ""
    st.write(response)

    with st.expander('Retrived Documents'): 
        st.info(retrieved_docs)

    with st.expander('Weather Information'): 
        st.info(get_rain_info())