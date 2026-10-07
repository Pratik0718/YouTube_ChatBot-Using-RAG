import os
import re
from urllib.parse import urlparse, parse_qs

import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="▶️", layout="wide")
st.title("▶️ YouTube RAG Chatbot")
st.write("Ask questions about a YouTube video's transcript using RAG and a Hugging Face model.")

def extract_video_id(url_or_id: str) -> str:
    value = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value

    parsed = urlparse(value)
    host = parsed.netloc.lower().split(":")[0]
    if host.endswith("youtu.be"):
        video_id = parsed.path.strip("/").split("/")[0]
    elif host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = [part for part in parsed.path.split("/") if part]
            video_id = (
                parts[1]
                if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live"}
                else ""
            )
    else:
        video_id = ""

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id or ""):
        raise ValueError("Please enter a valid YouTube URL or 11-character video ID.")
    return video_id


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


with st.sidebar:
    st.header("Hugging Face settings")
    model_id = st.text_input(
        "Model repository ID",
        value="deepseek-ai/DeepSeek-V3-0324",
        help="Use a model currently supported by an inference provider enabled for your account.",
    )
    token_from_env = os.environ.get("HF_TOKEN", "")
    token_from_secrets = ""
    try:
        token_from_secrets = st.secrets.get("HF_TOKEN", "")
    except Exception:
        pass
    hf_token = st.text_input(
        "Hugging Face token",
        value="",
        type="password",
        help="Alternatively configure HF_TOKEN as an environment variable or Streamlit secret.",
    )
    hf_token = hf_token.strip() or token_from_env or token_from_secrets
    st.caption("Keep your token private. Do not hardcode it or commit it to GitHub.")
    st.divider()
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=2000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=200, step=50)
    top_k = st.slider("Retrieved chunks (k)", min_value=1, max_value=8, value=4)

st.subheader("1. Choose a YouTube video")
with st.form("process_video_form"):
    video_input = st.text_input(
        "YouTube URL or video ID",
        placeholder="https://www.youtube.com/watch?v=Gfr50f6ZBvo",
    )
    process_video = st.form_submit_button("Fetch transcript and build index", type="primary")

if process_video:
    if not video_input.strip():
        st.warning("Enter a YouTube URL or video ID.")
    elif not hf_token:
        st.error("Enter your Hugging Face token in the sidebar, or configure HF_TOKEN.")
    else:
        try:
            video_id = extract_video_id(video_input)
            with st.spinner("Fetching English transcript..."):
                transcript_obj = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
                if hasattr(transcript_obj, "snippets"):
                    transcript = " ".join(
                        item.text for item in transcript_obj.snippets if item.text
                    )
                else:
                    transcript = " ".join(
                        item["text"] for item in transcript_obj.to_raw_data()
                        if item.get("text")
                    )

            if not transcript.strip():
                raise ValueError("The transcript is empty or unavailable.")

            with st.spinner("Splitting transcript and building the FAISS vector store..."):
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=int(chunk_size),
                    chunk_overlap=min(int(chunk_overlap), int(chunk_size) - 1),
                )
                chunks = splitter.create_documents([transcript])
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                vector_store = FAISS.from_documents(chunks, embeddings)
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": top_k},
                )

            llm = HuggingFaceEndpoint(
                repo_id=model_id.strip(),
                task="conversational",
                temperature=0.2,
                max_new_tokens=512,
                huggingfacehub_api_token=hf_token,
            )
            chat_model = ChatHuggingFace(llm=llm)

            # Test the model before building the RAG chain
            test_response = llm.invoke(
                "Reply with exactly: Model is working."
            )
            st.write("Model test:", repr(test_response.content))

            prompt = PromptTemplate(
                template=
                    """You are a helpful assistant. 
                       Answer only from the provided transcript context.
                       If the context is insufficient, say you don't know.

                    Transcript context:
                    {context}
                    
                    Question: {question}
                    
                    Answer:""",
                    input_variables=["context", "question"],
            )

            parallel_chain = RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            })
            parser = StrOutputParser()
            main_chain = parallel_chain | prompt | chat_model | StrOutputParser()

            st.session_state["video_id"] = video_id
            st.session_state["transcript"] = transcript
            st.session_state["vector_store"] = vector_store
            st.session_state["main_chain"] = main_chain
            st.session_state["chunk_count"] = len(chunks)
            st.session_state["model_id"] = model_id.strip()
            st.session_state["top_k"] = top_k
            st.success(
                f"Video indexed successfully: {len(transcript):,} transcript characters "
                f"split into {len(chunks)} chunks."
            )
        except TranscriptsDisabled:
            st.error("This video has captions disabled.")
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
            st.info(
                "If YouTube reports RequestBlocked, it may be blocking requests from your "
                "network or cloud IP. If Hugging Face reports model_not_supported or a task "
                "mismatch, select a model/provider supported by your account."
            )

if "main_chain" in st.session_state:
    st.divider()
    st.subheader("2. Ask questions about the video")
    st.caption(
        f"Indexed video ID: {st.session_state['video_id']} · "
        f"{st.session_state['chunk_count']} chunks · "
        f"Model: {st.session_state['model_id']}"
    )

    with st.form("question_form"):
        question = st.text_input(
            "Your question",
            placeholder="Can you summarize the video?",
        )
        ask = st.form_submit_button("Generate answer", type="primary")

    if ask:
        if not question.strip():
            st.warning("Enter a question.")
        else:
            try:
                with st.spinner("Retrieving transcript context and generating answer..."):
                    answer = st.session_state["main_chain"].invoke(question.strip())
                st.markdown("### Answer")
                st.write(answer if answer else "The model returned an empty answer. Try rephrasing or check model/provider support.")
                with st.expander("Inspect retrieved transcript chunks"):
                    docs = st.session_state["vector_store"].similarity_search(
                        question.strip(), k=st.session_state["top_k"]
                    )
                    for index, doc in enumerate(docs, start=1):
                        st.markdown(f"**Chunk {index}**")
                        st.write(doc.page_content)
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
                st.caption("Check the Hugging Face model/provider settings and token.")

elif not process_video:
    st.info("Paste a YouTube URL above and click **Fetch transcript and build index** to begin.")

st.divider()
st.caption("Built with Streamlit, LangChain, Hugging Face, Sentence Transformers, and FAISS.")
