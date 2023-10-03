import os
import pandas as pd
import evadb

cursor = evadb.connect().cursor()
print("Connected to EvaDB")

create_function_query = f"""CREATE FUNCTION IF NOT EXISTS ChatWithPandas
            IMPL  './functions/semantic_cache.py';
            """
cursor.query("DROP FUNCTION IF EXISTS ChatWithPandas;").execute()
cursor.query(create_function_query).execute()
print("Created Function")

create_table_query = f"""
CREATE TABLE IF NOT EXISTS CARSDATA(
id INTEGER,
name TEXT(30),
mpg INTEGER,
cyl FLOAT(64,64),
disp FLOAT(64,64),
hp FLOAT(64,64),
drat FLOAT(64,64),
wt FLOAT(64,64),
qsec FLOAT(64,64),
vs FLOAT(64,64),
am FLOAT(64,64),
gear FLOAT(64,64),
carb FLOAT(64,64)
);
"""
load_data_query = f""" LOAD CSV 'data/cars.csv' INTO CARSDATA;
"""

cursor.query(create_table_query).execute()
cursor.query(load_data_query).execute()
print("loaded data")

chat_query1 = f""" SELECT ChatWithPandas('what is the mean of the gear column',gear, name) FROM CARSDATA;
"""

result1 = cursor.query(chat_query1).execute()
print(result1)

chat_query2 = f""" SELECT ChatWithPandas('which car has the highest gear value',gear, name) FROM CARSDATA;
"""
result2 = cursor.query(chat_query2).execute()
print(result2)