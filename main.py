import os
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")

# Data ingestion into MongoDB vector database
import pandas as pd
from datasets import load_dataset

data = load_dataset("MongoDB/subset_arxiv_papers_with_embeddings")
dataset_df = pd.DataFrame(data["train"])

from pymongo import MongoClient

    # Initialize MongoDB python client
client = MongoClient(MONGO_URI)

DB_NAME = "agent_demo"
COLLECTION_NAME = "knowledge"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
collection = client.get_database(DB_NAME).get_collection(COLLECTION_NAME)

    # Delete any existing records in the collection
collection.delete_many({})

    # Data Ingestion
records = dataset_df.to_dict('records')
collection.insert_many(records)

print("Data ingestion into MongoDB completed")

# Create LangChain retriever with MongoDB
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=256)

    # Vector Store Creation
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGO_URI,
    namespace=DB_NAME + "." + COLLECTION_NAME,
    embedding= embedding_model,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    text_key="abstract"
    )

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Configure LLM using Fireworks AI
from langchain_openai import ChatOpenAI
from langchain_fireworks import Fireworks, ChatFireworks

llm = ChatFireworks(
    model="accounts/fireworks/models/firefunction-v1",
    max_tokens=256)

# Create tools for the agent
from langchain.agents import tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import ArxivLoader

@tool
def get_metadata_information_from_arxiv(word: str) -> list:
    """
    Fetches and returns metadata for a maximum of ten documents from arXiv matching the given query word.

    Args:
    word (str): The search query to find relevant documents on arXiv.

    Returns:
    list: Metadata about the documents matching the query.
    """
    docs = ArxivLoader(query=word, load_max_docs=10).load()
    # Extract just the metadata from each document
    metadata_list = [doc.metadata for doc in docs]
    return metadata_list

@tool
def get_information_from_arxiv(word: str) -> list:
    """
    Fetches and returns metadata for a single research paper from arXiv matching the given query word, which is the ID of the paper, for example: 704.0001.

    Args:
    word (str): The search query to find the relevant paper on arXiv using the ID.

    Returns:
    list: Data about the paper matching the query.
    """
    doc = ArxivLoader(query=word, load_max_docs=1).load()
    return doc

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="knowledge_base",
    description="This serves as the base knowledge source of the agent and contains some records of research papers from Arxiv. This tool is used as the first step for exploration and research efforts."
)

tools = [get_metadata_information_from_arxiv, get_information_from_arxiv, retriever_tool]

# Prompting the agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
agent_purpose = "You are a helpful research assistant"
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", agent_purpose),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ]
)

# Create the agent’s long-term memory using MongoDB
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain.memory import ConversationBufferMemory

def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(MONGO_URI, session_id, database_name=DB_NAME, collection_name="history")

memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=get_session_history("my-session")
)

# Agent creation
from langchain.agents import AgentExecutor, create_tool_calling_agent
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory,
)

# Agent execution
agent_executor.invoke({"input": "Get me a list of research papers on the topic Prompt Compression"})
agent_executor.invoke({"input":"Get me the abstract of the first paper on the list"})

