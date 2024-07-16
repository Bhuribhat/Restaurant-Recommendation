import os 
import requests
from dotenv import load_dotenv

from utils import read_file
from utils import embed_database

# Load environment variables from .env file
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# Inference Endpoint
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# Prompt Template
prompt = """You're a restaurant recommendation expert at บรรทัดทอง. 
No matter what the language of question you must answer in Thai.

Follow Up Input: {}

Knowledge: {}"""


def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


def inference_llama3(question):
    persist_directory = './vector_db'
    knowledges = read_file('./documents')
    vectordb = embed_database(documents=knowledges, persist_directory=persist_directory)

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"score_threshold": 0.6})
    retrieved_docs = retriever.get_relevant_documents(question)

    output = query({
        "inputs": prompt.format(question, retrieved_docs),
    })
    return output


if __name__ == '__main__':
    message = input("Enter a message: ").strip()
    output = inference_llama3(message)
    print(output)