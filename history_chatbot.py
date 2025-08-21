import os
import re
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from prompt import HISTORY_PROMPT_TEMPLATE

# Load environment variables
load_dotenv()

# Step 1: Embeddings and Vector Store (load if exists)
pdf_path = r"D:\ML\RAG\world_history.pdf"  # Safe Windows path
embeddings = TogetherEmbeddings(model="BAAI/bge-large-en-v1.5")

if os.path.isdir("faiss_index"):
    print("Loading existing FAISS index...")
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )
else:
    print("No FAISS index found. Creating new index...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("faiss_index")
    print("Saved FAISS index.")

# Step 2: LLM Setup (Together.ai)
llm = ChatTogether(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.3,
    max_tokens=800,
)

# Step 3: Load Prompt Template from prompt.py
def load_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        template=HISTORY_PROMPT_TEMPLATE,
        input_variables=["context", "chat_history", "question"],
    )

PROMPT = load_prompt_template()

# Step 4: Conversation Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
)

# Step 5: Conversational Retrieval Chain
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8},
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": PROMPT},
)

# Step 6: Chatbot Loop
def run_chatbot():
    print("History Chatbot ready! Type 'clear' to reset history, 'exit' to quit.")
    while True:
        query = input("Ask a history question: ")
        if not query:
            continue
        lowered = query.strip().lower()

        if lowered == 'exit':
            break
        if lowered == 'clear':
            memory.clear()
            print("Conversation history cleared.")
            continue

        # Check for greetings
        greeting_pattern = r"^(hi|hello|hey|my name is\s+([a-zA-Z]+)).*$"
        match = re.match(greeting_pattern, lowered)
        if match:
            if match.group(1).startswith("my name is"):
                name = match.group(2).capitalize()
            else:
                name = match.group(2).capitalize() if match.group(2) else "friend"
            print(f"Hello {name}, how can I assist you with history today?")
            continue

        result = qa_chain.invoke({"question": query})
        print("Answer:", result.get("answer", ""))
        try:
            sources = []
            for doc in result.get("source_documents", []):
                page = doc.metadata.get("page")
                sources.append(f"p{page + 1}" if isinstance(page, int) else "unknown")
            if sources:
                print("Sources:", ", ".join(sources))
        except Exception:
            pass

if __name__ == "__main__":
    run_chatbot()