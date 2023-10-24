import pandas as pd
import evadb

cursor = evadb.connect().cursor()
print("Connected to EvaDB")

create_function_query = f"""CREATE FUNCTION IF NOT EXISTS ChatWithPandas
            IMPL  './functions/chat_with_df.py';
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


columns = ['id', 'name', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']


load_data_query = f""" LOAD CSV 'data/cars.csv' INTO CARSDATA;""" 

cursor.query(create_table_query).execute()
cursor.query(load_data_query).execute()

print("loaded data")

# chat_query1 = f""" SELECT ChatWithPandas('query', 'what is the mean of the gear column',gear, name) FROM CARSDATA;
# """
# result1 = cursor.query(chat_query1).execute()
# print(result1)

# chat_query2 = f""" SELECT ChatWithPandas('plot', 'plot a bar graph of mpg vs hp',mpg, hp) FROM CARSDATA;
# """
# result2 = cursor.query(chat_query2).execute()
# print(result2)


# chat_query3 = f""" SELECT ChatWithPandas('manipulation', 'make all the names in the NAME column small', name) FROM CARSDATA;
# """
# result3 = cursor.query(chat_query3).execute()
# print(result3)

chat_query4 = f""" SELECT ChatWithPandas('cleaning',\
      'impute null values with average of the column if an integer or float. replace with an empty string if column is a string.\
        remove duplicate rows.', \
            id, name, mpg, cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb) FROM CARSDATA;
"""
result4 = cursor.query(chat_query4).execute()
print(result4)


