import json

from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseRetriever, Document

from utils import get_retriever


with open("data/info_schema.json", "r", encoding="utf-8") as f:
    schema = json.load(f)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
retriever = get_retriever(vector_db_directory = "./esi_vectorstore")

class MessagesState(MessagesState):
    question: str
    docs: list[Document]
    answer: str
    sql_query: str


def retrieve_node(state: MessagesState):
    RETRIEVER_PROMPT = "responde a la siguiente pregunta devolviendo todo el contexto relevante y mencionando las variables estadisticas relevantes mecionadas en la pregunta. {question}"
    question = state["messages"][-1].content

    docs = retriever.get_relevant_documents(RETRIEVER_PROMPT.format(question=question))
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
        "Eres un experto en economía con acceso al documento metodológico de una importante encuesta sobre los ingresos de la población chilena, donde se explican en detalle las decisiones metodológicas detrás de las variables en la base de datos. "
        "Responde la pregunta del usuario de manera clara, precisa y fundamentada, utilizando la información de las explicaciones metodológicas cuando sea pertinente."
    )
    
    prompt = f"{system_prompt}\n\nUser: {question}\nContext: {context}\nAssistant:"    

    # Generate answer
    answer = llm.invoke(prompt).content

    return {
        "question": question,
        "answer": answer,
    }

# Node: Generate SQL query from answer
def generate_sql_node(state: MessagesState):
    answer = state["answer"]
    question = state["question"]

    system_prompt = (
        "Eres un experto en bases de datos y SQL. "
        "Dada la siguiente respuesta a una pregunta sobre una base de datos de encuestas, "
        "genera una consulta SQL que permita obtener la información solicitada. "
        "Si la respuesta no es suficiente para generar una consulta SQL, responde solo con 'NO_SQL'."
    )
    prompt = f"{system_prompt}\n\nPregunta: {question}\nRespuesta: {answer}\nSQL:"

    sql_query = llm.invoke(prompt).content.strip()

    return {
        "question": question,
        "answer": answer,
        "sql_query": sql_query,
    }



def make_variable_selection_prompt(user_question: str, schema: dict) -> str:
    """
    Generate a prompt for an LLM to select the 10 most relevant variables from a schema
    based on a user question. The selected variables will be used for a SQL query.

    Args:
        user_question (str): The user's question about the data.
        schema (dict): The schema describing the columns/variables in the table.

    Returns:
        str: The prompt to send to the LLM.
    """
    # Format the schema as a readable list of variables with their descriptions, including nombre_codigo
    variable_descriptions = []
    for var, info in schema.items():
        label = info.get("Etiqueta", "")
        var_type = info.get("Tipo", "")
        valores = info.get("valores", {})
        nombre_codigo = valores.get("nombre_codigo", None)
        # Format nombre_codigo as a string, if present
        if nombre_codigo is not None:
            if isinstance(nombre_codigo, list):
                # Remove None values and join with "; "
                nombre_codigo_str = "; ".join([str(x) for x in nombre_codigo if x is not None])
            else:
                nombre_codigo_str = str(nombre_codigo)
            nombre_codigo_part = f", nombre_codigo: {nombre_codigo_str}"
        else:
            nombre_codigo_part = ""
        desc = f"{var} (Etiqueta: {label}, Tipo: {var_type}{nombre_codigo_part})"
        variable_descriptions.append(desc)
    schema_str = "\n".join(variable_descriptions)

    prompt = (
        "Eres un experto en análisis de datos y bases de datos. "
        "A continuación se presenta el esquema de una tabla, donde cada variable tiene un nombre, una etiqueta, un tipo y, si corresponde, los posibles valores de 'nombre_codigo'. "
        "Dada la siguiente pregunta del usuario, selecciona los 10 nombres de variables que sean más relevantes para responder la pregunta. "
        "Devuelve únicamente una lista de los nombres de las variables, separadas por comas, sin explicaciones adicionales.\n\n"
        f"Esquema de la tabla:\n{schema_str}\n\n"
        f"Pregunta del usuario: {user_question}\n\n"
        "Variables relevantes:"
    )
    return prompt





def get_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("retrieve", RunnableLambda(retrieve_node))
    graph.add_node("generate", RunnableLambda(generate_node))
    graph.add_node("generate_sql", RunnableLambda(generate_sql_node))  # Nuevo nodo agregado
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "generate_sql")  # Nueva transición al nuevo nodo
    graph.add_edge("generate_sql", END)         # El flujo termina en el nuevo nodo

    return graph.compile()
