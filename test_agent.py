import pandas as pd
import duckdb

ESI_DATA_FILE_PATH = 'esi-2023---personas.csv'
ESI_DF = pd.read_csv(ESI_DATA_FILE_PATH)
duckdb.sql(f"CREATE TABLE IF NOT EXISTS ESI AS SELECT * FROM ESI_DF")

def test_ingreso_promedio(sql_query):
    # Get the SQL result
    agent_average_income = duckdb.sql(sql_query).df()
    
    # Calculate average income
    OCUP_DF = ESI_DF.query('ocup_ref == 1')
    esi_average_income = (OCUP_DF['ing_t_p'] * OCUP_DF['fact_cal_esi']).sum() / OCUP_DF['fact_cal_esi'].sum()
    
assert abs(agent_average_income - esi_average_income) <= 0.01 * esi_average_income


def test_ingreso_promedio_mujeres(sql_query):
    # Get the SQL result
    agent_average_income_women = duckdb.sql(sql_query).df()
    
    # Calculate average income for women
    OCUP_DF = ESI_DF.query('ocup_ref == 1 and sexo == 2')
    esi_average_income_women = (OCUP_DF['ing_t_p'] * OCUP_DF['fact_cal_esi']).sum() / OCUP_DF['fact_cal_esi'].sum()
    
assert abs(agent_average_income_women - esi_average_income_women) <= 0.01 * esi_average_income_women


def test_ingreso_mediano(sql_query):
    # Get the SQL result
    agent_median_income = duckdb.sql(sql_query).df()
    
    # Calculate median income
    OCUP_DF = ESI_DF.query('ocup_ref == 1')
    df_sorted = OCUP_DF.sort_values('ing_t_p')
    df_sorted['cum_weights'] = df_sorted['fact_cal_esi'].cumsum()
    total_weight = df_sorted['fact_cal_esi'].sum()
    df_sorted['cum_weights_norm'] = df_sorted['cum_weights'] / total_weight
    median_position = df_sorted[df_sorted['cum_weights_norm'] >= 0.5].iloc[0]
    esi_median_income = median_position['ing_t_p']
    
assert abs(agent_median_income - esi_median_income) <= 0.01 * esi_median_income


def test_ingreso_p95(sql_query):
    # Get the SQL result
    agent_median_income = duckdb.sql(sql_query).df()
    
    # Calculate median income
    OCUP_DF = ESI_DF.query('ocup_ref == 1')
    df_sorted = OCUP_DF.sort_values('ing_t_p')
    df_sorted['cum_weights'] = df_sorted['fact_cal_esi'].cumsum()
    total_weight = df_sorted['fact_cal_esi'].sum()
    df_sorted['cum_weights_norm'] = df_sorted['cum_weights'] / total_weight
    p95_position = df_sorted[df_sorted['cum_weights_norm'] >= 0.95].iloc[0]
    esi_median_income = p95_position['ing_t_p']
    
assert abs(agent_median_income - esi_median_income) <= 0.01 * esi_median_income


def test_brecha_de_genero():
    pass