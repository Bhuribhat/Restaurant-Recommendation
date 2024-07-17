import os 
import requests
from utils import read_file
from utils import embed_database
from utils import get_rain_info
from utils import parse_source_document
from prompt import qa_prompt
from prompt import text_gen_prompt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# Model's Name
llama = "meta-llama/Meta-Llama-3-8B"
phi = "microsoft/Phi-3-mini-4k-instruct"
roberta = "deepset/roberta-base-squad2"
mt5 = "google/mt5-small"

# Inference Endpoint
API_URL = "https://api-inference.huggingface.co/models"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}


# Send payload to API Inference Endpoint
def query(payload, model_name: str):
    print(f"Using {model_name}")
    response = requests.post(f"{API_URL}/{model_name}", headers=headers, json=payload)
    return response.json()


# Retrieve information and inference LLM using RAG
def inference_huggingface(question: str, model_name: str=None):
    persist_directory = './vector_db'
    knowledges = read_file('./documents')
    vectordb = embed_database(documents=knowledges, persist_directory=persist_directory)

    # Load the contents of the documents to vertor database
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"score_threshold": 0.6})
    retrieved_docs = retriever.get_relevant_documents(question)
    retrieved_docs = parse_source_document(retrieved_docs)

    # Get rain information
    rain_info = get_rain_info()

    # Get the response (Default model is mt5)
    if model_name is None:
        model_name = mt5

    # Question-Answering Model
    if model_name == roberta:
        input_prompt = qa_prompt.format(question, rain_info)
        output = query({
            "inputs": {
                "question": input_prompt,
                "context": retrieved_docs,
            }
        }, model_name)

    # Text-Completion Model
    elif model_name in [llama, phi, mt5]:
        input_prompt = text_gen_prompt.format(question, rain_info, retrieved_docs)
        input_length = len(input_prompt)
        output = query({
            "inputs": input_prompt,
        }, model_name)

    # Return response text and retrived documents
    if 'error' in output:
        return output['error'], 'Error occurs'
    else:
        return output['answer'] if model_name == roberta else output[0]['generated_text'][input_length:], retrieved_docs


if __name__ == '__main__':
    message = input("Enter a message: ").strip()
    output = inference_huggingface(message)
    print(output)