# `Banthat Thong` Restaurant Recommendation

## Overall & Use case

Suppose you have some text documents (PDF, blog, Notion pages, etc.) and want to ask questions related to the contents of those documents. LLMs, given their proficiency in understanding text, are a great tool for this.

The Restaurant Recommender System leverages the power of Hugging Face and OpenAI library to provide intelligent and contextual restaurant recommendations. This system is designed to answer user queries about restaurant recommendations in Thai, incorporating real-time weather information such as rain status to enhance the user experience.

`Tools:` Streamlit, LangChain, FastAPI, Chroma, FAISS, Huggingface, and OpenAI

## How does Embedding work

The pipeline for converting raw unstructured data into vector database.

- Loading: First we need to load our data. Unstructured data can be loaded from many sources. Use the `LangChain integration hub` to browse the full set of loaders. Each loader returns data as a `LangChain Document`.
- Splitting: `Text splitters` break Documents into splits of specified size
- Storage: Storage (e.g., often a `vectorstore`) will house `and often embed` the splits
- Retrieval: The app retrieves splits from storage (e.g., often with `similar embeddings` to the input question)

## Hugging Face Integration

This section provides instructions on how to integrate Hugging Face models for generating restaurant recommendations.

### Overview

We use inference endpoint https://api-inference.huggingface.co/models to create a conversational AI that recommends restaurants. The model uses a predefined prompt template to ensure responses are concise and in Thai.

`Models:` Meta-Llama-3-8B, roberta-base-squad2, Phi-3-mini-4k-instruct

### Define the Prompt Template

Create a prompt template that will guide the model to provide restaurant recommendations and include additional information like rain status.

```py
prompt = """You are a restaurant recommendation expert at บรรทัดทอง.
Use the provided contents to answer the question, and also inform the user about the rain.
Please provide a short and concise response.
Regardless of the language of the question, you must answer in Thai.

Rain Information: {rain_info}

Question: {question}"""
```

### Function: `inference_huggingface`

This function uses the Hugging Face inference endpoint to generate responses based on the input message, retrived documents, and rain information.

```py
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

    # Get the response (Default model is roberta)
    if model_name is None:
        model_name = roberta

    # Question-Answering Model
    if model_name == roberta:
        output = query({
            "inputs": {
                "question": qa_prompt.format(question, rain_info),
                "context": retrieved_docs,
            }
        }, model_name)

    # Text-Completion Model
    elif model_name in [llama, phi]:
        output = query({
            "inputs": text_gen_prompt.format(question, rain_info, retrieved_docs),
        }, model_name)

    # Return response text and retrived documents
    if 'error' in output:
        return output['error'], 'Error occurs'
    else:
        return output['answer'] if model_name == roberta else output[0]['generated_text'], retrieved_docs
```

## External API Usage

This section describes how to use the [Open-Meteo](https://open-meteo.com/) API to extract rain information for the Banthat Thong location, then the information is sent to LLM for more suggestion.

### Overview

The Open-Meteo API provides weather forecast data for various locations around the world. This guide explains how to use the API to determine if it is about to rain at the Banthat Thong location.

### Prerequisites

- Python 3.x
- `requests` library

### Function: `get_rain_info`

This function makes a request to the Open-Meteo API to retrieve current weather data, including precipitation, for the Banthat Thong location. It then determines whether rain is expected based on the precipitation data.

```py
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
```

### Usage

The function handles HTTP request errors by checking the status code of the response. If the request fails, it prints an error message with the status code and returns `None`.

```py
# Example usage of the get_rain_info function
rain_info = get_rain_info()
print(rain_info)
```

> __Note:__ The API request is made using the GET method with specific parameters for latitude, longitude, and weather data types (temperature, wind speed, and precipitation). The function checks for the presence of precipitation data and returns appropriate messages based on the data.

## How to run MVP (Minimum Viable Product)

### Setup Environment in `.env` File

```bash
OPENAI_API_KEY="OPENAI_API_KEY"
HUGGINGFACE_API_KEY="HUGGINGFACE_API_KEY"
```

### Create Conda Environment

```bash
# Create & Activate environment
$ conda create -n llm-inference python=3.9
$ conda activate llm-inference

# Install dependacies
$ pip install -r requirements.txt
```

### Running Chatbot

```bash
streamlit run main.py
```

<!-- TODO find new model for better QA -->
<!-- https://github.com/Bhuribhat/Programming-Project/blob/main/Question_Answering.ipynb -->
<!-- https://docs.google.com/document/d/1qlnm3tsF_WjCp8iESiTX3Ttdon2ZG11bvBhE3YUhW4s/edit -->   