import os
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "your key"
os.environ["OPENAI_API_BASE"] = "url"


class LLMs:
    # 用gpt4来测
    def __init__(self, model="gpt-4-turbo-2024-04-09", request_type="openai", parameters={"top_p": 0.7, "temperature": 0.9}):
        self.model = model
        self.request_type = request_type

        assert request_type == "openai"
        
        self.client = ChatOpenAI(model_name=model)
        self.client.model_kwargs = parameters

    def request(self, prompt, sys_prompt=''):
        try:
            batch_messages = [[
                SystemMessage(content=sys_prompt),
                HumanMessage(content=prompt),
            ]]
            results = self.client.generate(batch_messages)
            model_output = results.generations[0][0].text
            return model_output
        except Exception as e:
            print(e)
            return None

if __name__ == "__main__":
    pass