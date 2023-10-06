
import pandas as pd

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.functions.gpu_compatible import GPUCompatible

from datastructure.aidDataframe import AIDataFrame

class ChatWithPandas(AbstractFunction):


    @setup(cacheable=False, function_type="FeatureExtraction", batchable=False)
    def setup(self, use_local_llm=False, local_llm_model=None):
        self.use_local_llm = use_local_llm
        self.local_llm_model = local_llm_model
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
    def forward(self, df: pd.DataFrame):
        query = df[0][0]
        req_df = df.drop([0], axis=1)
        smart_df = AIDataFrame(req_df, description="A dataframe about cars")
        if self.use_local_llm:
            smart_df.initialize_local_llm_model(local_llm=self.local_llm_model)
            prompt = f"""There is a dataframe in pandas (python). This is the result of print(req_df.head()):\n
            {str(req_df.head())}. Answer to the following question: {query}."""
            print("PROMPTT", prompt)
            response = smart_df.chat(prompt, local=self.use_local_llm)
            script = response.split("```")[1]
            # script = response
            load_df = f"import pandas as pd\ndf = pd.read_csv('/home/preethi/projects/pandas-ai-integration/data/cars.csv')\n"
            print(load_df + "\n" + script)
            ans = load_df + "\n" + script
            print("ANSWERRR/n", ans)
        else:
            smart_df.initialize_middleware()
            response = smart_df.chat(query, local=self.use_local_llm)
        df_dict = {"response": [response]}
        
        ans_df = pd.DataFrame(df_dict)
        return pd.DataFrame(ans_df)
    
