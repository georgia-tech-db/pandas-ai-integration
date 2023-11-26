from gpt4all import GPT4All
import pandas as pd
import openai
from config import Config
import re
import os


class AIDataFrame(pd.DataFrame):
    def __init__(self, df, config=None, description=None, name=None) -> None:
        super().__init__(df)

        #initialize pandas dataframe
        self.pd_df = df
        self.config = Config()
        
        if len(df)>0:
            self.is_df_loaded = True
        else:
            self.is_df_loaded = False

        #set the description
        self.description = description
        
        #set the config
        if config:
            self.config = config
        
        #set name
        self.name = name

    
    @property
    def col_count(self):
        if self.is_df_loaded:
            return len(list(self.pd_df.columns))
        
    @property
    def row_count(self):
        if self.is_df_loaded:
            return len(self.pd_df)
        
    @property
    def sample_head_csv(self):
        if self.is_df_loaded:
            return self.pd_df.head(5).to_csv()
        
    
    @property
    def metadata(self):
        return self.pd_df.info()
    
    def to_csv(self, file_path):
        self.pd_df.to_csv(file_path)
    
        
    def initialize_middleware(self):
        """ Initializes openai api with the openai key and model """

        open_ai_key = self.config.get_open_ai_key()
        openai.api_key = open_ai_key
        self.openai_model = "gpt-3.5-turbo"
        return
    
    def create_prompt(self, query):
        prompt = f"""
        I need you to write a python3.8 program for the following dataframe.
        You are given the following pandas dataframe.
        The dataframe has {self.col_count} columns. The columns are {list(self.columns)}. 
        The datatypes of the columns are {list(self.pd_df.dtypes)}.
        The first 2 rows of data in the csv format are {self.iloc[0:2].to_csv()} .
        Write the python code for the following query is {query}. 
        Write this code in a function named 'pandas_function' and it should take the pandas dataframe as input. 
        Give below is the python code template. Create a copy of the original dataframe and perform the operations on this dataframe. Fill in the required code as per the query.

        ```
        def pandas_function(pd_df):
            df_copy = pd_df.copy()

            #if remove duplicates, select the column then use the df.drop_duplicates
            #if calculating any value first select the column then apply the required operation
            #if performing a plot, select the x and y axes and then create the plot.
            #to deal with outliers, update the data in df_copy.
            #to impute null values perform operations column-wise
                #for replacing null values in string column, use df_copy[['str_column']].fillna(''). don't use np.object. use object.
                #for replacing null values in int or float columns use df_copy[['int_column]].fillna(df_copy['int_column'].mean())

            #if the query results in updates to the dataframe
            return df_copy

            #elif query results in a single value
            #val = df['col1'].mean()
            #return val

        ```
        If the query results in a new dataframe, save it to a file called new_df.csv. Return the string "dataframe saved to new_df.csv".
        If the query requests a plot, then save the plot to file called plot.png. Return the string "plot saved to plot.png".
        If the final result is a single value, just return the value.
        Add the required imports for the function. 
        Do not add any code for example usage to execute the function. Write only the function code.
        The response should have only the python code and no additional text. 
        I repeat.. give the python code only for the function. NO ADDITIONAL CODE. 
        Do not add any additional identifier in the beginning to indicate that it's python code."
        """
        prompt = re.sub(' +', ' ', prompt)
        return prompt


    def execute_python(self, python_code: str):
        """
         A function to execute the python code and return result. 
         
         Args
         python_code - the python code to be executed. It is in string format.
         type - type of function to be executed. query | plot | manipulation
         
         Returns
         Return the result of the execution.
        """

        with open("tmp.py", "w") as file:
            file.write(python_code)
        
        from tmp import pandas_function
        result = pandas_function(self.pd_df)

        os.remove("tmp.py")
        return result
    
    def query_dataframe(self, query: str, custom: bool = False):
        
        prompt = ""
        
        if not custom:
            prompt = self.create_prompt(query)
        else:
            #if the entire prompt is given by the user
            prompt = query
        
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", \
                                                  temperature=0.2, \
                                                  messages=[{"role": "user", "content": prompt}])
        
        python_code = completion.choices[0].message.content
        result = self.execute_python(python_code)
    
        return result

    #     prompt = self.create_data_cleaning_prompt(clean_instructions)

    #     completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", \
    #                                               temperature=0.2, \
    #                                               messages=[{"role": "user", "content": prompt}])

    #     python_code = completion.choices[0].message.content
    #     answer = self.execute_python(python_code, "data_cleaning")
    #     return answer
    
    def query_localgpt(self, query: str, local_llm_model: str):
        prompt = self.create_prompt(query)   
        local_llm = GPT4All(local_llm_model)
        response = local_llm.generate(prompt)
        if "```" in response:
            python_code = response.split("```")[1].lstrip("python")
        else:
            python_code = response
        result = self.execute_python(python_code)
        return result

    def general_clean_dataframe(self):
        prompt = f"""I need you to write a python3.8 program for the following dataframe. 
            You are given the following pandas dataframe. 
            The dataframe has {self.col_count} columns. The columns are {list(self.columns)}. 
            The first 2 rows of data in the csv format are {self.iloc[0:2].to_csv()} .
            Give me the python code to perform the following data cleaning: 
            Replace null values in integer or float type columns with the mean of that column. 
            Replace null values in string columns with empty string. 
            Replace values in integer or float type columns that are greater than 2 standard deviations from mean with the mean value of that column.
            Remove the duplicate values.
            Write this code in a function named 'pandas_function' and it should take the pandas dataframe as input. output should be the cleaned dataframe.
            Give below is the python code template. Create a copy of the original dataframe and perform the operations on this dataframe. 
            Fill in the required code as per the requirement.

            ```
            def pandas_function(pd_df):
                df_copy = pd_df.copy()

                #if remove duplicates, select the column then use the df.drop_duplicates
                #if calculating any value first select the column then apply the required operation
                #if performing a plot, select the x and y axes and then create the plot.
                #to deal with outliers, update the data in df_copy.
                #for replacing null values in string column, use df_copy[['str_column']].fillna(''). don't use np.object. use object.
                #for replacing null values in int or float columns use df_copy[['int_column]].fillna(df_copy['int_column'].mean())

                #if the query results in updates to the dataframe
                return df_copy

                #elif query results in a single value
                #val = df['col1'].mean()
                #return val

            ```
            Add the required imports for the function. 
            Do not add any code for example usage to execute the function. Write only the function code.
            The response should have only the python code and no additional text.
            I repeat.. give the python code only for the function. NO ADDITIONAL CODE."
            Do not add any additional identifier in the beginning to indicate that it's python code."
            """

        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", \
                                                  temperature=0.2, \
                                                  messages=[{"role": "user", "content": prompt}])
        
        python_code = completion.choices[0].message.content
        answer = self.execute_python(python_code)
        return answer




    
    
    


