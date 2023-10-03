
import pandas as pd

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.functions.gpu_compatible import GPUCompatible

from datastructure.aidDataframe import AIDataFrame

class ChatWithPandas(AbstractFunction):


    @setup(cacheable=False, function_type="FeatureExtraction", batchable=False)
    def setup(self):
        pass

    @property
    def name(self) -> str:
        return "SentenceTransformerFeatureExtractor"

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["data"],
                column_types=[NdArrayType.STR],
                column_shapes=[(1)],
            ),

        ],
        output_signatures=[
            PandasDataframe(
                columns=["response"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(1, 384)],
            )
        ],
    )
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        
        query = df[0][0]
        req_df = df.drop([0], axis=1)
        
        smart_df = AIDataFrame(req_df, description="A dataframe about cars")
        smart_df.initialize_middleware()

        response = smart_df.chat(query)
        
        df_dict = {"response": [response]}
        
        ans_df = pd.DataFrame(df_dict)
        return pd.DataFrame(ans_df)

