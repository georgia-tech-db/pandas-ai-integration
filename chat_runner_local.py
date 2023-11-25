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
          IMPL  './functions/chat_with_df.py' use_local_llm 'True' local_llm_model "llama-2-7b-chat.ggmlv3.q4_0.bin" csv_path {csv_file};
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

#custom prompt
prompt = "Write a python3.8 program for the following Pandas dataframe. The columns are title, genres and budget.\
     Calculate the mean value of the budget column. Write the code in a function called pandas_function that takes as input the dataframe.\
     return the result in a string format. Do not create examples.\
     Use the given dataframe. Do not add any code for example usage to execute the function. Write only the function code."

chat_query5 = f"""SELECT ChatWithPandas('custom prompt', '{prompt}', title, genres, budget) 
              FROM sqlite_data.DUPL_DATA;"""

result5 = cursor.query(chat_query5).execute()
print(result5)