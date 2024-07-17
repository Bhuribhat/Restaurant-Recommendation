import os 
import torch
import requests

from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


# Indexing and splitting the contents of the documents
def read_file(file_path: str) -> list[Document]:
    documents = os.listdir(file_path)
    text_splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=200)
    loaders = []
    for file_name in documents:
        each_file_path = os.path.join(file_path, file_name)
        text_loader = TextLoader(each_file_path, encoding='utf8')
        loaders += text_splitter.split_text(text_loader.load()[0].page_content)
    knowledges = [Document(page_content=loader) for loader in loaders]
    return knowledges


# Load Embedding Model
def load_embedding_model(embedding_model_name):
    if torch.cuda.is_available():
        device_type = "cuda"
    else:
        device_type = "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device_type},
    )
    return embeddings


# Load the contents of the documents to vertor database
def embed_database(
    documents,
    persist_directory,
    vector_store="faiss",
):
    embeddings = load_embedding_model("intfloat/multilingual-e5-small")

    # Embedding if exists
    if os.path.isdir(persist_directory):
        if vector_store == "faiss":
            vectordb = FAISS.load_local(
                persist_directory,
                embeddings
            )
        elif vector_store == "chroma":
            vectordb = Chroma(
                embedding_function=embeddings, 
                persist_directory=persist_directory
            )
            vectordb.persist()
        else:
            raise NotImplementedError(
                f"Embedding Algorithm {vector_store} is not supported/implemented"
            )

    # Create embeddings if not exists
    else:
        if vector_store == "faiss":
            vectordb = FAISS.from_documents(
                documents=documents,
                embedding=embeddings,
            )
            vectordb.save_local(persist_directory)
        elif vector_store == "chroma":
            vectordb = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=persist_directory,
            )
            vectordb.persist()
        else:
            raise NotImplementedError(
                f"Embedding Algorithm {vector_store} is not supported/implemented"
            )
    return vectordb


# Post-processing the retrieved documents
def parse_source_document(source_docs, n_chunks: int=1) -> str:
    if source_docs is not None:
        contents = []
        page_count = 0
        for page in source_docs:
            if page_count < n_chunks:
                contents.append(page.page_content)
                page_count += 1
        result = "\n\n".join(contents)
        return result
    else:
        return ""


# Extract rain information at Banthat Thong Location from https://open-meteo.com/
def get_rain_info():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 13.74,
        "longitude": 100.52,
        "current": "temperature_2m,wind_speed_10m,precipitation"
    }

    # Make a GET request to the API
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()

        # Extract precipitation value
        precipitation = data.get('current', {}).get('precipitation', None)

        # Determine if rain is about to fall
        if precipitation is not None:
            if precipitation > 0:
                return "Rain is about to fall."
            else:
                return "No rain is expected soon."
        else:
            return "Precipitation data not available."
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return None