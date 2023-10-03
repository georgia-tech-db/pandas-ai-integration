class ErrorCorrectionPrompt:
    def __init__(self) -> None:
        pass

    def get_prompt(self, df, *argv):
        user_question = argv[0]
        error = argv[1]
        python_code = argv[2]

        text = f"""
        Given the following pandas dataframe {df}, and the following question was asked {user_question}.
        The following python code was generated : {python_code}.
        This code gave the following error: {error}.
        Correct the python code and return a new python code (do not import anything) that fixes the above mentioned error. 
        Do not generate the same code again.
        """
        return text