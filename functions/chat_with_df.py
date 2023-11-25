
import pandas as pd
import os

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe


from datastructure.aidDataframe import AIDataFrame

class ChatWithPandas(AbstractFunction):

    @setup(cacheable=False, function_type="FeatureExtraction", batchable=False)
    def setup(self):
        pass

    @setup(cacheable=False, function_type="FeatureExtraction", batchable=False)
    def setup(self, use_local_llm=False, local_llm_model=None, csv_path=None):
        self.use_local_llm = use_local_llm
        self.local_llm_model = local_llm_model
        # self.csv_path = csv_path
        pass

    @property
    def name(self) -> str:
        return "ChatWithPandas"

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["data"],
                column_types=[NdArrayType.STR],
                column_shapes=[(None, 3)],
            ),

        ],
        output_signatures=[
            PandasDataframe(
                columns=["response"],
                column_types=[NdArrayType.STR],
                column_shapes=[(None,)],
            )
        ],
    )
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:

        path = os.getcwd()
        os.chdir(path=path)

        query = df.iloc[0,0]
        
        if query != "custom prompt":
            req_df = df.drop([0], axis=1)
            smart_df = AIDataFrame(req_df)
            smart_df.initialize_middleware()

            if query == "general cleaning":
                cleaned_df = smart_df.general_clean_dataframe()
                response = "cleaned dataframe is saved to cleaned_df.csv. The following steps were done\
                    1. Replaced null values with mean of column or empty string.\
                    2. Replaced outliers with the mean of the column"
                    
                #save to a csv file
                cleaned_df.to_csv("cleaned_df.csv")
            else:
                response = smart_df.query_dataframe(query)
        
        elif query == "custom prompt":
            custom_query = df.iloc[0,1]
            req_df = df.drop([0], axis=1)
            smart_df = AIDataFrame(req_df)
            if not self.use_local_llm:
                smart_df.initialize_middleware()
                response = smart_df.query_dataframe(custom_query, custom=True)
            else:
                response = smart_df.query_localgpt(custom_query, self.local_llm_model, custom=True)
        df_dict = {"response": [response]}
        
        ans_df = pd.DataFrame(df_dict)
        return ans_df

