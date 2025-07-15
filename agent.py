from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseRetriever, Document

from utils import get_retriever

llm = ChatOpenAI(model="gpt-4o", temperature=0)
retriever = get_retriever(vector_db_directory = "./esi_vectorstore")

class MessagesState(MessagesState):
    question: str
    docs: list[Document]
    answer: str


def retrieve_node(state: MessagesState):
    question = state["messages"][-1].content
    docs = retriever.get_relevant_documents(question)
    return {
        "question": question,
        "docs": docs,
    }

# Node: Generate answer from LLM
def generate_node(state: MessagesState):
    question = state["question"]
    docs: list[Document] = state["docs"]

    # Combine context and history
    context = "\n\n".join(doc.page_content for doc in docs)

    system_prompt = (
        "You are a biblical scholar. Answer the user's question clearly, "
        "and at the end, include a short parable in the style of the Bible that summarizes the answer "
        "in an illustrative way. Use simple, reverent language.")
    
    prompt = f"{system_prompt}\n\nUser: {question}\nContext: {context}\nAssistant:"    

    # Generate answer
    answer = llm.invoke(prompt).content

    return {
        "question": question,
        "answer": answer,
    }


def get_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("retrieve", RunnableLambda(retrieve_node))
    graph.add_node("generate", RunnableLambda(generate_node))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()