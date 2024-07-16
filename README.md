# `Banthat Thong` Restaurant Recommendation

`Tools:` LangChain, FastAPI, Chroma, and OpenAI

## Setup Environment in `.env` File

```bash
OPENAI_API_KEY="OPENAI_API_KEY"
HUGGINGFACE_API_KEY="HUGGINGFACE_API_KEY"
```

## Create Conda Environment

```
# Create & Activate environment
$ conda create -n llm-inference python=3.9
$ conda activate llm-inference

# Install dependacies
$ pip install -r requirements.txt
```

## Running Chatbot

```bash
python main.py
```