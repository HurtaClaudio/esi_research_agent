from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseRetriever

def get_retriever(vector_db_directory: str = '') -> BaseRetriever:
    vectorstore = Chroma(
        persist_directory=vector_db_directory,
        embedding_function=OpenAIEmbeddings()
    )
    return vectorstore.as_retriever()