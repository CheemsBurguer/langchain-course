import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

#Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#Iniiallize vector store
vector_store = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
)

#Initialize chat model
model = init_chat_model("gpt-3.5-turbo", model_provider="openai")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about Halo Lore"""
    # Retrieve top 4 most simiar documents
    retrieved_docs = vector_store.as_retriever().invoke(query, k=4)

    # Seriallize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    # Return both serialized content and raw documents
    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    
    Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The user's question

    Returns:
    Dictionary containing:
        -answer: The generated answer
        -context: List of retrieved documents

    """

    # Create the agent with retrieval tool

    system_prompt = (
        "You are a helpul AI assistant that answers questions about Halo Lore."
        "You have access to a tool that retrieves revelant documentaion."
        "Use the tool to find relevant information before nswering questions."
        "Always cite the resources you use in youtr answerds."
        "If you can not find the anser in the retrieved documentation, say so."
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # Build message list

    messages = [{"role": "user", "content": query}]

    # Invoke the agent
    response = agent.invoke({"messages": messages})

    # Extract the answer from the agent last AI agent message

    answer = response["messages"][-1].content

    # Extract context documents form ToolMessage artifacts
    context_docs = []

    for message in response['messages']:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, 'artifact'):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == "__main__":
    result = run_llm("Los precursores tienen relacion con el gravemind?")
    print(result)