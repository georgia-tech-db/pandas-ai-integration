import unittest
import os
import pandas as pd
import evadb

class TestEvaDBFunctions(unittest.TestCase):

    def setUp(self):
        self.conn = evadb.connect()
        self.cursor = self.conn.cursor()
        print("Connected to EvaDB")

        create_function_query = f"""CREATE FUNCTION IF NOT EXISTS ChatWithPandas
                                    IMPL  './functions/semantic_cache.py' use_local_llm 'True' local_llm_model "llama-2-7b-chat.ggmlv3.q4_0.bin";
                                    """
        self.cursor.query("DROP FUNCTION IF EXISTS ChatWithPandas;").execute()
        self.cursor.query(create_function_query).execute()
        print("Created Function")

        create_table_query = """
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
        load_data_query = """ LOAD CSV 'data/cars.csv' INTO CARSDATA;
                            """

        self.cursor.query(create_table_query).execute()
        self.cursor.query(load_data_query).execute()
        print("Loaded data")

    def test_mean_of_gear_column(self):
        chat_query = "SELECT ChatWithPandas('what is the mean of the gear column', gear, name) FROM CARSDATA;"
        result = self.cursor.query(chat_query).execute()
        print("RESULTT-", result)
        self.assertIsNotNone(result)

    def test_highest_gear_value_car(self):
        chat_query = "SELECT ChatWithPandas('which car has the highest gear value', gear, name) FROM CARSDATA;"
        result = self.cursor.query(chat_query).execute()
        print("RESULTTT2: ", result)
        self.assertIsNotNone(result)

    def tearDown(self):
        self.cursor.close()
        print("Closed EvaDB connection")

if __name__ == '__main__':
    unittest.main()
