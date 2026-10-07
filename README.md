# YouTube RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about a YouTube video's transcript. It uses LangChain to build the pipeline, Hugging Face for embeddings and chat-model inference, and FAISS for vector search.

## Features

- Fetches English captions with `youtube-transcript-api`
- Splits transcript text into overlapping chunks
- Creates embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Stores and retrieves relevant chunks with FAISS
- Uses a Hugging Face-hosted chat model to answer questions from retrieved transcript context
- Includes a LangChain chain for asking follow-up questions

## Tech stack

- Python
- Google Colab or Jupyter Notebook
- LangChain
- Hugging Face Hub / Inference Providers
- Sentence Transformers
- FAISS
- YouTube Transcript API

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/youtube-rag-chatbot.git
cd youtube-rag-chatbot
```

Replace `YOUR_USERNAME` with your GitHub username and adjust the repository name if needed.

### 2. Install dependencies

In a notebook, run:

```python
%pip install -r requirements.txt
```

Alternatively, in a terminal:

```bash
python -m pip install -r requirements.txt
```

### 3. Set your Hugging Face token

Create a Hugging Face access token at https://huggingface.co/settings/tokens. Make sure your token has the permissions needed for inference with the provider/model you choose.

For Google Colab, enter the token without hardcoding it in the notebook:

```python
from getpass import getpass
import os

os.environ["HF_TOKEN"] = getpass("Enter your Hugging Face token: ")
```

Never commit your token, paste it into source code, or include it in notebook outputs. If a token is exposed, revoke it and create a replacement.

### 4. Choose a YouTube video

Set `video_id` to the video ID only (not the full URL):

```python
video_id = "YOUR_VIDEO_ID"
```

The video must be accessible and have captions available in a language you request.

### 5. Configure the Hugging Face model

The notebook uses `HuggingFaceEndpoint` and `ChatHuggingFace`. Set `repo_id` to a model that is currently supported by an inference provider enabled for your account, and set the task to match that provider's supported task. Provider availability and pricing can change; check https://huggingface.co/models?inference_provider=all.

Example structure:

```python
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os

llm = HuggingFaceEndpoint(
    repo_id="YOUR_SUPPORTED_MODEL_ID",
    task="conversational",  # Match the selected model/provider's supported task
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
    max_new_tokens=512,
    temperature=0.2,
)

chat_model = ChatHuggingFace(llm=llm)
```

### 6. Ask questions

After the transcript has been fetched, chunked, embedded, and indexed, invoke the RAG chain:

```python
answer = main_chain.invoke("Can you summarize the video?")
print(answer)
```

The chatbot is instructed to answer using the retrieved transcript context and to say when the context is insufficient.

## How it works

1. **Transcript extraction:** fetches captions for the selected YouTube video.
2. **Text splitting:** divides the transcript into chunks (the notebook uses a chunk size of 1,000 characters and overlap of 200).
3. **Embeddings:** converts chunks into vectors using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Vector store:** stores vectors in FAISS.
5. **Retrieval:** retrieves the four most relevant chunks for a question.
6. **Generation:** sends the retrieved context and question to the Hugging Face chat model.
7. **Answer parsing:** returns the model's response as text.

## Troubleshooting

- **YouTube `RequestBlocked`:** YouTube may block requests from cloud-hosted notebook IPs, including Colab. This is not necessarily a code error. Try again later, run locally, or provide a transcript obtained through YouTube's available transcript UI or another authorized source. Do not repeatedly retry in a loop.
- **Captions unavailable:** the video may have captions disabled or may not provide captions in the requested language.
- **Hugging Face `model_not_supported` or task mismatch:** choose a model/provider combination supported by your account and use the task specified by the provider. An open-source model does not guarantee free hosted inference.
- **Missing token:** set `HF_TOKEN` in the notebook session before invoking the model.
- **Empty or poor answers:** inspect the retrieved chunks and confirm they contain information relevant to the question.

## Project structure

```text
youtube-rag-chatbot/
├── Youtube_Chatbot.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Security

Do not commit Hugging Face access tokens, API keys, cookies, or other credentials. Review notebook code and outputs before publishing the repository.

## License

Add a license file if you want to specify how others may use, modify, and distribute this project.
