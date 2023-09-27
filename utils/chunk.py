import pandas as pd
from evadb.catalog.catalog_type import ColumnType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe

class Chunk(AbstractFunction):
    """
    Arguments:
        None

    Input Signatures:
        input_dataframe (DataFrame) : A DataFrame containing a column of strings.

    Output Signatures:
        output_dataframe (DataFrame) : A DataFrame containing chunks of strings.

    Example Usage:
        You can use this function to concatenate strings in a DataFrame and split them into chunks.
    """

    @property
    def name(self) -> str:
        return "Chunk"

    @setup(cacheable=False)
    def setup(self) -> None:
        # Any setup or initialization can be done here if needed
        pass

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["text"],
                column_types=[ColumnType.TEXT],
                column_shapes=[(None,)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["chunks"],
                column_types=[ColumnType.TEXT],
                column_shapes=[(None,)],
            )
        ],
    )
    def forward(self, input_dataframe):
        # Ensure input is provided
        if input_dataframe.empty:
            raise ValueError("Input DataFrame must not be empty.")

        # Define the maximum number of tokens per chunk
        max_tokens_per_chunk = 100  # Adjust this value as needed

        # Initialize lists for the output DataFrame
        output_strings = []

        # Iterate over rows of the input DataFrame
        for _, row in input_dataframe.iterrows():
            input_string = row["text"]

            # Split the input string into chunks of maximum tokens
            chunks = [input_string[i:i + max_tokens_per_chunk] for i in range(0, len(input_string), max_tokens_per_chunk)]

            output_strings.extend(chunks)

        # Create a DataFrame with the output strings
        output_dataframe = pd.DataFrame({"chunks": output_strings})

        return output_dataframe