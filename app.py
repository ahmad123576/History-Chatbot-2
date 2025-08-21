import os
import asyncio
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from prompt import HISTORY_PROMPT_TEMPLATE

load_dotenv()

st.set_page_config(page_title="History RAG Chatbot", page_icon="📜", layout="wide")

# Ensure an event loop exists in Streamlit thread
def _ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

@st.cache_resource(show_spinner=False)
def get_vectorstore(pdf_path: str, _embeddings):
    index_dir = "faiss_index"
    if os.path.isdir(index_dir):
        return FAISS.load_local(index_dir, _embeddings, allow_dangerous_deserialization=True)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    vs = FAISS.from_documents(chunks, _embeddings)
    vs.save_local(index_dir)
    return vs

@st.cache_resource(show_spinner=False)
def build_chain(pdf_path: str, model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1", temperature: float = 0.3, k: int = 8):
    _ensure_event_loop()
    embeddings = TogetherEmbeddings(model="BAAI/bge-large-en-v1.5")
    vectorstore = get_vectorstore(pdf_path, embeddings)

    llm = ChatTogether(model=model_name, temperature=temperature, max_tokens=800)
    prompt = PromptTemplate(template=HISTORY_PROMPT_TEMPLATE, input_variables=["context", "chat_history", "question"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return chain

# Sidebar
st.sidebar.title("📜 History RAG Chatbot")
pdf_path_default = os.path.abspath("world_history.pdf")
pdf_path = st.sidebar.text_input("PDF path", value=pdf_path_default)
model_name = st.sidebar.selectbox("Model", ["mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Llama-3.1-70B-Instruct"], index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
k = st.sidebar.slider("Retriever k", 3, 12, 8, 1)
rebuild = st.sidebar.button("Rebuild Index")
clear_chat = st.sidebar.button("Clear Chat History")

if rebuild:
    try:
        if os.path.isdir("faiss_index"):
            for name in os.listdir("faiss_index"):
                try:
                    os.remove(os.path.join("faiss_index", name))
                except Exception:
                    pass
            try:
                os.rmdir("faiss_index")
            except Exception:
                pass
        st.cache_resource.clear()
        st.success("Index cleared. It will be rebuilt on next use.")
    except Exception as e:
        st.error(f"Failed to clear index: {e}")

# Build / get chain
if not os.path.exists(pdf_path):
    st.error("PDF not found. Please set a valid path in the sidebar.")
    st.stop()

try:
    _ensure_event_loop()
    qa_chain = build_chain(pdf_path, model_name=model_name, temperature=temperature, k=k)
except Exception as e:
    st.error(f"Failed to initialize chain: {e}")
    st.stop()

if clear_chat:
    try:
        qa_chain.memory.clear()
        st.success("Chat history cleared.")
    except Exception:
        pass

# Chat Interface
st.title("History RAG Chatbot")
st.caption("Ask questions grounded in your PDF. The bot answers only from the document context.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a history question...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Check for greetings
        greeting_pattern = r"^(hi|hello|hey|my name is\s+([a-zA-Z]+)).*$"
        match = re.match(greeting_pattern, user_input.strip().lower())
        if match:
            if match.group(1).startswith("my name is"):
                name = match.group(2).capitalize()
            else:
                name = match.group(2).capitalize() if match.group(2) else "friend"
            answer = f"Hello {name}, how can I assist you with history today?"
            st.markdown(answer)
        else:
            with st.spinner("Thinking..."):
                try:
                    result = qa_chain.invoke({"question": user_input})
                    answer = result.get("answer") or result.get("result") or result.get("output_text") or ""
                    st.markdown(answer)

                    # Sources expander
                    src_docs = result.get("source_documents", [])
                    if src_docs:
                        with st.expander("Sources"):
                            for i, doc in enumerate(src_docs, start=1):
                                pg = doc.metadata.get("page")
                                page_label = f"p{pg + 1}" if isinstance(pg, int) else "unknown"
                                snippet = doc.page_content[:300].replace("\n", " ")
                                st.markdown(f"**Source {i} ({page_label})**: {snippet}...")
                except Exception as e:
                    st.error(str(e))
                    answer = ""

        st.session_state.messages.append({"role": "assistant", "content": answer})

# Footer
st.write("")
st.caption("Built with LangChain + Together.ai + FAISS. Your data stays local.")