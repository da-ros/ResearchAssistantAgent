# Building an AI Agent With Memory Using MongoDB, Fireworks AI, and LangChain

This project demonstrates how to build an AI agent with long-term memory capabilities using MongoDB Atlas, Fireworks AI, and LangChain. The agent can perform complex tasks by leveraging a knowledge base, interactively responding to queries, and maintaining context over multiple interactions.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Data Ingestion](#data-ingestion)
- [Creating the AI Agent](#creating-the-ai-agent)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

The goal of this project is to build an AI research assistant capable of understanding and responding to queries by utilizing an extensive knowledge base. The agent's memory is powered by MongoDB Atlas, with the LangChain framework providing the structure for language models, and Fireworks AI enabling advanced language processing.

## Prerequisites

Before starting, ensure you have the following:
- Python 3.8 or higher
- MongoDB Atlas account and cluster
- Fireworks AI and OpenAI API keys
- Internet connection for downloading datasets

## Setup and Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/da-ros/ResearchAssistantAgent.git
    cd ResearchAssistantAgent
    ```

2. **Create a Virtual Environment and Install Dependencies:**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Set Up Environment Variables:**
    Create a `.env` file in the root directory of the project and add your API keys and MongoDB URI:

    ```plaintext
    FIREWORKS_API_KEY=your_fireworks_api_key
    OPENAI_API_KEY=your_openai_api_key
    MONGO_URI=your_mongo_db_uri
    ```

## Data Ingestion

The first step involves ingesting data into a MongoDB database. This data will be used to build the knowledge base for the AI agent.

- **Load and Ingest Data:**
    The dataset from Hugging Face's repository is used to populate the MongoDB collection. This dataset includes research papers with embeddings.

## Creating the AI Agent

The core functionality of the AI agent includes creating a retriever with LangChain, setting up a language model with Fireworks AI, and configuring tools for querying and processing information.

1. **LangChain Retriever:**
    A retriever is created using LangChain to facilitate similarity searches within the MongoDB database.

2. **Fireworks AI Language Model:**
    The language model is configured using Fireworks AI to handle natural language processing tasks and function calling functionality.

3. **Tool Creation:**
    Tools are defined to fetch metadata and information from arXiv, integrating with the LangChain framework.

4. **Long-term Memory Setup:**
    MongoDB is used to maintain the agent's conversation history, allowing it to remember and utilize previous interactions.

## Usage

Run the main script to start the AI agent and interact with it:

```bash
python main.py
```

## Example queries:

    "Get me a list of research papers on the topic Prompt Compression."
    "Get me the abstract of the first paper on the list."

These queries demonstrate the agent's ability to retrieve and process information based on the provided knowledge base and context.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

    MongoDB
    Fireworks AI
    LangChain

By leveraging MongoDB Atlas for scalable data storage, Fireworks AI for advanced language modeling, and LangChain for structured language processing, this project showcases a comprehensive approach to building intelligent AI agents with memory capabilities.