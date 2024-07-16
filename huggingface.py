import os 
import requests
from utils import read_file
from utils import embed_database
from utils import parse_source_document
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# Model's Name
llama = "meta-llama/Meta-Llama-3-8B"
phi = "microsoft/Phi-3-mini-4k-instruct"

# Inference Endpoint
API_URL = "https://api-inference.huggingface.co/models"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# Prompt Template
prompt = """You're a restaurant recommendation expert at บรรทัดทอง.
You can use contents provided in knowledge section to answer the question.
No matter what the language of question you must answer in Thai.

Question:\n{}

Knowledge:\n{}"""


def query(payload, model_name: str):
    if model_name in [llama, phi]:
        response = requests.post(f"{API_URL}/{model_name}", headers=headers, json=payload)
    else:
        response = requests.post(f"{API_URL}/{llama}", headers=headers, json=payload)
    return response.json()


def inference_huggingface(question: str, model_name: str):
    persist_directory = './vector_db'
    knowledges = read_file('./documents')
    vectordb = embed_database(documents=knowledges, persist_directory=persist_directory)

    # Load the contents of the documents to vertor database
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"score_threshold": 0.6})
    retrieved_docs = retriever.get_relevant_documents(question)
    retrieved_docs = parse_source_document(retrieved_docs)

    # Get the response
    output = query(
        {
            "inputs": prompt.format(question, retrieved_docs),
        }, model_name
    )
    return output[0]['generated_text']


if __name__ == '__main__':
    message = input("Enter a message: ").strip()
    output = inference_huggingface(message)
    print(output)