from img2table.document import PDF
import os
import pandas as pd


def extract_raw_tables():
    pdf_path = "data/personas-esi-2023-manual-guia-de-variables.pdf"

    output_dir = "extracted_tables"
    os.makedirs(output_dir, exist_ok=True)

    pdf_doc = PDF(pdf_path)

    extracted_tables = pdf_doc.extract_tables()

    return extracted_tables


def extract_info_schema(master_df):
    info_schema = {}
    for _, row in master_df.iterrows():

        var_name = str(row.get("Variable"))
        if var_name:
            last_non_null_var_name = var_name
            info_schema[var_name] = {
                "Etiqueta": row.get("Etiqueta"),
                "Tipo": row.get("Caract."),
                "valores": {"codigo": [row.get("Cód.")],
                "nombre_codigo": [row.get("Nombre Código")]}
            }
        
        # Si el nombre de la variable es nulo, entonces es una fila de valores. Agregar codigos y nombres de codigo a la ultima variable no nula.
        else:
            info_schema[last_non_null_var_name]["valores"]["codigo"].append(row.get("Cód."))
            info_schema[last_non_null_var_name]["valores"]["nombre_codigo"].append(row.get("Nombre Código"))

    return info_schema


def get_master_df(extracted_tables):
    master_df = pd.DataFrame()

    for i in range(5, 33):
        df = extracted_tables[i][0].df
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        master_df = pd.concat([master_df, df])

    return master_df


def extract_tables_from_ESI_glossary():
    extracted_tables = extract_raw_tables()
    master_df = get_master_df(extracted_tables)
    info_schema = extract_info_schema(master_df)
    return info_schema


if __name__ == "__main__":
    info_schema = extract_tables_from_ESI_glossary()
    print(info_schema)