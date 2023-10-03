class Config:
    def __init__(self) -> None:
        self.open_ai_key = ""
        self.local_llm_model = "llama-2-7b-chat.ggmlv3.q4_0.bin"

    def get_open_ai_key(self):
        return self.open_ai_key
    def get_local_llm_model(self):
        return self.local_llm_model
    