
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

        query = df.iloc[0,1]
        type = df.iloc[0,0]
        
        req_df = df.drop([0,0], axis=1)
        
        smart_df = AIDataFrame(req_df, description="A dataframe about cars")
        smart_df.initialize_middleware()

        if type == "cleaning":
            cleaned_df = smart_df.clean_dataframe(query)
            #save to a csv file
            cleaned_df.to_csv("cleaned_df.csv")
            response = "cleaned dataframe is saved to cleaned_df.csv"
        
        if type == "query":
            response = smart_df.query_dataframe(query)
        elif type == "plot":
            response = smart_df.plot_dataframe(query)
        elif type == "manipulation":
            response = smart_df.manipulate_dataframe(query)
        
        
        df_dict = {"response": [response]}
        
        ans_df = pd.DataFrame(df_dict)
        return pd.DataFrame(ans_df)

