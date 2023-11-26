import pandas as pd
import evadb
import sqlite3

# csv_file = '/home/preethi/Downloads/data/Movie/duplicates/dirty_train.csv'
csv_file = '/home/preethi/projects/splprobs/pandas-ai-integration/data/cars.csv'
cursor = evadb.connect().cursor()
print("Connected to EvaDB")

#Drop function if it already exists
cursor.query("DROP FUNCTION IF EXISTS ChatWithPandas;").execute()

#create a new function
create_function_query = f"""CREATE FUNCTION IF NOT EXISTS ChatWithPandas
          IMPL  './functions/chat_with_df.py' use_local_llm 'True' local_llm_model "gpt4all-falcon-q4_0.gguf" csv_path "{csv_file}";
          """
cursor.query(create_function_query).execute()
print("Created Function")

#create table in sqlite and connect to evadb
sql_db = """CREATE DATABASE IF NOT EXISTS sqlite_data WITH ENGINE = 'sqlite', PARAMETERS = {
     "database": "evadb.db"
};"""

cursor.query(sql_db).execute()

#load data into sqlite
df = pd.read_csv(csv_file)
database_file = 'evadb.db'
conn = sqlite3.connect(database_file)
table_name = 'DUPL_DATA'
df.to_sql(table_name, conn, if_exists='replace', index=False)
conn.commit()
conn.close()
print("Loaded data")

#query
chat_query2 = f""" SELECT ChatWithPandas('custom prompt', 'How many unique car names are present?',
            id, name, mpg, cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb) FROM sqlite_data.DUPL_DATA;
"""
result2 = cursor.query(chat_query2).execute()
print(result2)
