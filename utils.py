import os 
import torch

from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS


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


# TODO
def parse_source_document(source_docs):
    pass