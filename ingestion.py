import asyncio
import os, ssl
from typing import Any, Dict, List
import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUEST_CA_BUNCLE"] = certifi.where()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=True, chunk_size=50, retry_min_seconds=10
)

vectorstore = PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"), embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""

    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"VectorStore Indexing: Preparing to add {len(documents)} documents to vectore store"
    )

    # Create batches

    batches = [
        documents[i: i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.add_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfuly added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num}/{len(batches)} - {e}")
            return False
        return True
    
    #Process batches concurrently
    tasks = [add_batch(batch, i) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count succesfuly batches

    succesful = sum(1 for result in results if result is True)

    if succesful == len(batches):
        log_success(
            f"VextorStore Indexing: All batches processed succesfully ({succesful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Proccessed {succesful}/{len(batches)} batches succesfully"
        )

    

async def main():
    """Main function"""

    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        " TavilyCrawl: Starting to Crawl documentation from https://www.halopedia.org/",
        Colors.PURPLE
    )

    res = tavily_crawl.invoke({
        "url": "https://www.halopedia.org/",
        "max_depth": 5,
        "extract_depth": "advanced",
        "instructions": "content on Precursors"
    })

    all_docs = [Document(page_content=result["raw_content"], metadata={"source": result['url']}) for result in res['results'] if result["raw_content"]]
    log_success(
        f"TavilyCrawl: Succesfully crawles {len(all_docs)} URLs from site"
    )

    log_header("DOCUMENT CHUNKING PHASE")

    log_info(
        f"Text Spitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)

    splitted_docs = text_splitter.split_documents(all_docs)

    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    #Process documents asynchroniously

    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETED")
    log_success("Documentation ingestion pipeline finished sucessfully")

if __name__ == "__main__":
    asyncio.run(main())