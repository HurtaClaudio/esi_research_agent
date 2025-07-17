import json

from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseRetriever, Document

from utils import get_retriever


with open("data/info_SCHEMA.json", "r", encoding="utf-8") as f:
    SCHEMA = json.load(f)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
retriever = get_retriever(vector_db_directory = "./esi_vectorstore")

class MessagesState(MessagesState):
    question: str
    docs: list[Document]
    sql_query: str
    selected_variables: list


def retrieve_node(state: MessagesState):
    RETRIEVER_PROMPT = "responde a la siguiente pregunta devolviendo todo el contexto relevante y mencionando las variables estadisticas relevantes mecionadas en la pregunta. {question}"
    question = state["messages"][-1].content

    docs = retriever.get_relevant_documents(RETRIEVER_PROMPT.format(question=question))
    return {
        "question": question,
        "docs": docs,
    }


# Node: Generate SQL query from answer
def generate_sql_node(state: MessagesState):
    docs = state["docs"]
    question = state["question"]
    selected_variables = state.get("selected_variables", [])

    # Extract information about selected variables from SCHEMA
    variables_context = []
    for var in selected_variables:
        var = var.strip()
        if var in SCHEMA:
            var_info = SCHEMA[var]
            label = var_info.get("Etiqueta", "")
            var_type = var_info.get("Tipo", "")
            valores = var_info.get("valores", {})
            codigo = valores.get("codigo", None)
            descripcion_codigo = valores.get("nombre_codigo", None)
            
            var_desc = f"Nombre variable:{var} (Descripcion variable: {label}, Tipo: {var_type})"
            if codigo:
                var_desc += f", nombre codigo: {codigo} - descripcion codigo: {descripcion_codigo}"
            variables_context.append(var_desc)
    
    variables_context_str = "\n".join(variables_context)

    # Extract content from retrieved documents
    docs_content = "\n".join([doc.page_content for doc in docs])

    system_prompt = (
        "Eres un experto en bases de datos y SQL. "
        "Dada la siguiente información relevante sobre una base de datos de encuestas, "
        "genera una consulta SQL que permita obtener la información solicitada en la pregunta. "
        "Si la información no es suficiente para generar una consulta SQL, responde solo con 'NO_SQL'."
        "Solo puedes usar las variables proveídas como parte de tu respuesta."
        f"\n\nVariables relevantes seleccionadas:\n{variables_context_str}"
        f"\n\nInformación relevante:\n{docs_content}"
    )
    prompt = f"{system_prompt}\n\nPregunta: {question}\nSQL:"

    print("SQL generation prompt:")
    print(prompt)

    sql_query = llm.invoke(prompt).content.strip()

    return {
        "question": question,
        "docs": docs,
        "sql_query": sql_query,
    }
def variable_selection_node(state: MessagesState):
    """
    Generate a prompt for an LLM to select the 10 most relevant variables from a SCHEMA
    based on a user question. The selected variables will be used for a SQL query.
    Returns:
        str: The prompt to send to the LLM.
    """
    # Format the SCHEMA as a readable list of variables with their descriptions, including nombre_codigo
    variable_descriptions = []
    for var, info in SCHEMA.items():
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
        f"Pregunta del usuario: {state["question"]}\n\n"
        "Variables relevantes:"
    )

    selected_variables = llm.invoke(prompt).content.split(",")
    
    return {"selected_variables": selected_variables}



def get_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("retrieve", RunnableLambda(retrieve_node))
    graph.add_node("variable_selection", RunnableLambda(variable_selection_node))
    graph.add_node("generate_sql", RunnableLambda(generate_sql_node))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "variable_selection")
    graph.add_edge("variable_selection", "generate_sql")
    graph.add_edge("generate_sql", END)
    return graph.compile()
