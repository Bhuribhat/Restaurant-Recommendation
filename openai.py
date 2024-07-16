import os 
from dotenv import load_dotenv
from utils import read_file

from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Prompt Template
prompt = """You're a restaurant recommendation expert at บรรทัดทอง. 
No matter what the language of question you must answer in Thai.

Follow Up Input: {question}"""


# Load the contents of the documents to vertor database
def create_vectordb(knowledges: list[Document], file_path: str, load_from_file: bool=False) -> Chroma:
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    if load_from_file == True:
        vectorstore = Chroma(persist_directory=file_path, embedding_function=embedding)
    else:
        vectorstore = Chroma.from_documents(documents=knowledges, embedding=embedding, persist_directory=file_path)
    return vectorstore


# Retrieve information and inference LLM using RAG
def inference_gpt(message: str) -> str:
    knowledges = read_file('./documents')
    vectorstore = create_vectordb(
        knowledges,
        file_path="./vector_db", 
        load_from_file=True
    )

    # Load QA model
    prompt_template = PromptTemplate.from_template(prompt)
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-16k") 

    # Create chain for question answering
    retriever = vectorstore.as_retriever()
    conversation = ConversationalRetrievalChain.from_llm(
        llm=chat, 
        retriever=retriever, 
        condense_question_prompt=prompt_template
    )

    # Get the response
    response = conversation(message)
    print(response)
    return response["answer"]