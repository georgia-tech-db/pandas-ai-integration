import pandas as pd
import numpy as np
import openai
import subprocess
from gpt4all import GPT4All
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from prompts.error_correction_prompt import ErrorCorrectionPrompt
from config import Config


class AIDataFrame(pd.DataFrame):
    def __init__(self, df, config=None, description=None, name=None) -> None:
        super().__init__(df)

        #initialize pandas dataframe
        self.pd_df = df
        print("pd_df INITTT: \n", str(self.pd_df))
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
        if name:
            self.name = name

        #setup cache
        self.cache = {}

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
    

    def clear_cache(self):
        self.cache = {}
    
        
    def initialize_middleware(self):
        open_ai_key = self.config.get_open_ai_key()

        self.llm_agent = create_pandas_dataframe_agent(OpenAI(temperature=0, openai_api_key=open_ai_key), \
                                        self.pd_df, verbose=False)
        openai.api_key = open_ai_key
        self.openai_model = "text-davinci-003"
        return
    
    def initialize_local_llm_model(self, local_llm=None):
        if local_llm:
            local_llm_model = local_llm
        else:
            local_llm_model = self.config.get_local_llm_model(local_llm_model)
        self.local_llm = GPT4All(local_llm_model)
        return

    def query_dataframe(self, query):
        if query not in self.cache:
            ans = self.llm_agent.run(query)
            self.cache[query] = ans
        else:
            ans= self.cache[query]
        return ans
    
    def code_error_correction(self, query, error, old_python_code):
        prompt = ErrorCorrectionPrompt().get_prompt(self.pd_df, query, error, old_python_code)
        #print(prompt)
        response = openai.Completion.create(engine = self.openai_model, prompt = prompt)
        answer = response.choices[0].text

        return answer

    def chat(self, prompt, local=False):
        if local:
            # print("QUERYYY QQQQ", query)
            # query = f"""There is a dataframe in pandas (python). The name of the
            # dataframe is self.pd_df. This is the result of print(self.pd_df):\n
            # {str(self.pd_df.head())}. Return a python script without any other text to get the answer to the following question: {prompt}. Do not write code to load the CSV file."""
            # print("QUERYYY QQQQ", query)
            # query = f"""There is a dataframe in pandas (python). The name of the
            # dataframe is self.pd_df. This is the result of print(self.pd_df):\n
            # {str(self.pd_df.head())}. Please provide a Python script without any introductory text answer the question: {prompt}. Do not write code to load the CSV file."""
            
            # query = f"""There is a dataframe in pandas (python). The name of the
            # dataframe is self.pd_df. This is the result of print(self.pd_df.head()):
            # {str(self.pd_df.head())}. Return a python script without any other text to the following question: {prompt}. Do not write code to load the CSV file."""
            return self.local_llm.generate(prompt)
        
        else:
            return self.llm_agent.run(prompt)

        
