import os 
from dotenv import load_dotenv
from utils import get_weather_info
from utils import embed_database
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
prompt = """You are a restaurant recommendation expert at บรรทัดทอง.
Use the provided contents to answer the question, and also inform user about the rain.
Please provide a short and concise response.
Regardless of the language of the question, you must answer in Thai.

Rain Information: {rain_info}

Question: {question}"""


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
    rain_info = get_weather_info()
    knowledges = read_file('./documents')
    vectorstore = create_vectordb(
        knowledges,
        file_path="./vector_db", 
        load_from_file=True
    )

    # Load the GPT model
    prompt_template = PromptTemplate.from_template(prompt.format(rain_info=rain_info))
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-0125") 

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


if __name__ == '__main__':
    message = input("Enter a message: ").strip()
    output = inference_gpt(message)
    print(output)