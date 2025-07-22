import json
import pandas as pd
import duckdb

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

ESI_DATA_FILE_PATH = 'data/esi-2023---personas.csv'
ESI_DF = pd.read_csv(ESI_DATA_FILE_PATH)


llm = ChatOpenAI(model="gpt-4o", temperature=0)
retriever = get_retriever(vector_db_directory = "./esi_vectorstore")

class MessagesState(MessagesState):
    question: str
    docs: list[Document]
    sql_query: str
    selected_variables: list
    extracted_data: pd.DataFrame
    summary_statistics: any
    summary_code: str



def retrieve_node(state: MessagesState):
    question = state["messages"][-1].content
    RETRIEVER_PROMPT = f"""responde a la siguiente pregunta devolviendo todo el contexto relevante y mencionando las variables estadisticas relevantes mecionadas en la pregunta del usuario.
    Si la pregunta del usuario es muy generica, tambíen devuelve contexto que pueda ayudar a responder de manera mas específica la pregunta de acuerdo a las mejores práticas descritas en el documento.
    Pregunta del usuario: {question}
    """

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
            
            var_desc = f"Nombre variable: {var} (Descripcion variable: {label}, Tipo: {var_type})"
            if codigo:
                var_desc += f", nombre codigo: {codigo} - descripcion codigo: {descripcion_codigo}"
            variables_context.append(var_desc)
    
    variables_context_str = "\n".join(variables_context)

    # Extract content from retrieved documents
    docs_content = "\n".join([doc.page_content for doc in docs])

    system_prompt = (
        "Eres un experto en bases de datos y SQL. "
        "Tienes información relevante extraída de el documento metodologico de la generacion de la encuesta."
        "Tambien tienes acceso al schema de datos de algunas variables relevantes seleccionadas para tí."
        "genera una consulta SQL que permita obtener la información para responder a la pregunta del usuario. "
        "Si la información no es suficiente para generar una consulta SQL, responde solo con 'NO_SQL'."
        "Solo puedes usar las variables {selected_variables} como parte de tu respuesta."
        "Si hay varias formas de obtener la misma información, selecciona la forma mas sencilla."
        f"El nombre de la tabla es 'ESI'."
        f"\n\nVariables relevantes seleccionadas:\n{variables_context_str}"
        f"\n\nInformación relevante:\n{docs_content}"
        
    )
    prompt = f"{system_prompt}\n\nPregunta: {question}\nSQL:"

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
    selected_variables = [var.strip() for var in selected_variables]
    
    return {"selected_variables": selected_variables}

def lookup_data_node(state: MessagesState):
    """Implementation of sales data lookup from Dataframe file using SQL"""
    try:

        duckdb.sql(f"CREATE TABLE IF NOT EXISTS ESI AS SELECT * FROM ESI_DF")

        sql_query = state["sql_query"].strip()
        sql_query = sql_query.replace("```sql", "").replace("```", "")
        
        result = duckdb.sql(sql_query).df()
        return {"extracted_data": result}

    except Exception as e:
        print(f"An Exception error occurred: {str(e)}")
        print("An empty DataFrame is returned")
        return {"extracted_data": pd.DataFrame()}


def execute_code_node(state: MessagesState):
    question = state["question"]
    df = state["extracted_data"]
    selected_variables = state["selected_variables"]

    system_prompt = f"""
    Eres un experto en análisis de datos con pandas. Genera código Python que cree las estadísticas
    solicitadas en la pregunta del usuario y el DataFrame proporcionado. El código debe:
    1. Usar el DataFrame 'df' que ya está disponible
    2. Crear la estadísticas solicitadas en la pregunta del usuario
    3. No generar estadisticas que no sean relevantes para la pregunta.
    4. Solo se pueden usar las variables {selected_variables} en el análisis.
    
    Solo devuelve el código Python, sin explicaciones adicionales.
    """

    prompt = f"{system_prompt}\n\nPregunta del usuario: {question}\n\nDataFrame disponible:\n{df.head(3)}\n\nCódigo Python:"
    
    summary_code = llm.invoke(prompt).content.strip()
    summary_code = summary_code.replace("```python", "").replace("```", "")
    summary_code = summary_code.strip()
    print(f"summary_code is {summary_code}")
    
    # Execute the generated code to produce summary statistics
    try:
        local_vars = {"df": df}
        exec(summary_code, globals(), local_vars)
        summary_result = local_vars.get("result", "No se generaron resultados.")
    except Exception as e:
        summary_result = f"Error al ejecutar el código: {str(e)}"
    
    return {
        "summary_statistics": summary_result,
        "summary_code": summary_code
    }



def get_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("retrieve", RunnableLambda(retrieve_node))
    graph.add_node("variable_selection", RunnableLambda(variable_selection_node))
    graph.add_node("generate_sql", RunnableLambda(generate_sql_node))
    graph.add_node("lookup_data", RunnableLambda(lookup_data_node))
    graph.add_node("execute_code", RunnableLambda(execute_code_node))


    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "variable_selection")
    graph.add_edge("variable_selection", "generate_sql")
    graph.add_edge("generate_sql", "lookup_data")
    graph.add_edge("lookup_data", "execute_code")    
    graph.add_edge("execute_code", END)
    return graph.compile()
