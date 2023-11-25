import pandas as pd
import evadb
import sqlite3

cursor = evadb.connect().cursor()
print("Connected to EvaDB")

#Drop function if it already exists
cursor.query("DROP FUNCTION IF EXISTS ChatWithPandas;").execute()

#create a new function
create_function_query = f"""CREATE FUNCTION IF NOT EXISTS ChatWithPandas
            IMPL  './functions/chat_with_df.py';
            """

cursor.query(create_function_query).execute()
print("Created Function")

#create table in sqlite and connect to evadb
sql_db = """CREATE DATABASE IF NOT EXISTS sqlite_data WITH ENGINE = 'sqlite', PARAMETERS = {
     "database": "evadb.db"
};"""

cursor.query(sql_db).execute()

#load data into sqlite
csv_file = 'clean_ml_data/Movie/duplicates/dirty_train.csv'
df = pd.read_csv(csv_file)
database_file = 'evadb.db'
conn = sqlite3.connect(database_file)
table_name = 'DUPL_DATA'
df.to_sql(table_name, conn, if_exists='replace', index=False)
conn.commit()
conn.close()
print("Loaded data")

#using the generic cleaning function without any specific instruction
chat_query_1 = f""" SELECT ChatWithPandas('general cleaning',
            title, genres, budget, language, duration, year, vote_count, score) FROM sqlite_data.DUPL_DATA;
"""
result = cursor.query(chat_query_1).execute()
print(result)

#performing a calculation
chat_query2 = f""" SELECT ChatWithPandas('what is the mean budget',
            title, genres, budget, language, duration, year, vote_count, score) FROM sqlite_data.DUPL_DATA;
"""
result2 = cursor.query(chat_query2).execute()
print(result2)

#plotting
chat_query3 = f""" SELECT ChatWithPandas('plot a chart of year vs budget',
            title, genres, budget, language, duration, year, vote_count, score) FROM sqlite_data.DUPL_DATA;
"""
result3 = cursor.query(chat_query3).execute()
print(result3)

#data manipulation
chat_query4 = f""" SELECT ChatWithPandas('make all the names in the title column small', title) 
              FROM sqlite_data.DUPL_DATA;
"""
result4 = cursor.query(chat_query4).execute()
print(result4)


#custom prompt
prompt = "Write a python3.8 program for the following Pandas dataframe. The columns are title, genres and budget.\
     Calculate the mean value of the budget column. Write the code in a function called pandas_function that takes as input the dataframe.\
     return the result in a string format. Do not create examples.\
     Use the given dataframe. Do not add any code for example usage to execute the function. Write only the function code."

chat_query5 = f"""SELECT ChatWithPandas('custom prompt', '{prompt}', title, genres, budget) 
              FROM sqlite_data.DUPL_DATA;"""

result5 = cursor.query(chat_query5).execute()
print(result5)