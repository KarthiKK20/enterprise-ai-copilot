from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ------------------------
# 1. LLM
# ------------------------

llm = OllamaLLM(
    model="llama3",
    temperature=0,
    base_url="http://127.0.0.1:11434"
)

# ------------------------
# 2. Load + Split Docs
# ------------------------

loader = TextLoader("documents/sample.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# ------------------------
# 3. Embeddings + FAISS
# ------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434"
)

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# ------------------------
# 4. Memory Store (Session Based)
# ------------------------

conversation_memory = {}

def get_memory(session_id: str):
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
    return conversation_memory[session_id]

# ------------------------
# 5. Prompt with Memory
# ------------------------

prompt = ChatPromptTemplate.from_template(
    """
    You are an enterprise AI assistant.

    Conversation History:
    {history}

    Context:
    {context}

    Question:
    {question}

    Answer clearly and concisely.
    """
)

# ------------------------
# 6. RAG Chain
# ------------------------

from operator import itemgetter

rag_chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "history": itemgetter("history")
    }
    | prompt
    | llm
    | StrOutputParser()
)

def ask_question(query: str, session_id: str):
    history = get_memory(session_id)

    formatted_history = "\n".join(
        [f"{role}: {msg}" for role, msg in history]
    )

    response = rag_chain.invoke({
        "question": query,
        "history": formatted_history
    })

    history.append(("User", query))
    history.append(("Assistant", response))

    return response