from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.knowledge.langchain import LangChainKnowledgeBase
import os
from dotenv import load_dotenv
load_dotenv()


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

pc = Pinecone()
index = pc.Index(name=os.environ["PINECONE_INDEX_NAME"])
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

loader = PyPDFLoader(file_path="1700079.pdf")
docs = loader.load_and_split(
    RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
)

vector_store.add_documents(docs)
retriever = vector_store.as_retriever()

knowledge_base = LangChainKnowledgeBase(retriever=retriever)

agent = Agent(
    model=Ollama(id="llama3.2"),
    knowledge=knowledge_base,
    name="RAG Agent",
    description="You are a RAG agent that answers based on retrieved documents",
    instructions="Answer user query based on retrieved documents",
    search_knowledge=True,
    stream=True
)

agent.print_response("Who is the main character in this document and tell me about that character", stream=True)