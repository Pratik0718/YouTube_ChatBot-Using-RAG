# YouTube RAG Chatbot — Streamlit UI

This project wraps the notebook's YouTube transcript RAG workflow in a simple Streamlit interface.

## Features
- Accepts a YouTube URL or video ID
- Fetches available English captions
- Splits transcript into overlapping chunks
- Embeds chunks using `sentence-transformers/all-MiniLM-L6-v2`
- Stores and retrieves chunks with FAISS
- Builds the LangChain RAG chain
- Calls a Hugging Face-hosted chat model
- Displays answers and retrieved context

## Run locally

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:
```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Hugging Face token
Enter your token in the app sidebar, or set `HF_TOKEN` as an environment variable. Do not commit tokens to GitHub. If a token was exposed, revoke it and create a new one.

## Model compatibility
The notebook's model ID `Qwen/Qwen3.8-27B` is preserved as the initial value from the provided code, but hosted provider support may change. If Hugging Face returns `model_not_supported` or a task mismatch, choose a model/provider currently supported by your account and match the expected task. Open-source model weights do not guarantee free hosted inference.

## Known limitation
YouTube may block transcript requests from cloud IPs, and some videos do not have English captions. These are external availability issues and are not necessarily bugs in the UI.

## Project files
- `Youtube_Chatbot.ipynb`: original notebook
- `app.py`: Streamlit user interface and RAG workflow
- `requirements.txt`: Python dependencies
