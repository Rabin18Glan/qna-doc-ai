from langchain_core.runnables import Runnable, RunnableMap
from chat_models.chat_prompts import format_prompt
import streamlit as st
class Chain(Runnable):
    def __init__(self, search, tagger, structured_model, chat_model):
        self.search = search
        self.tagger = tagger
        self.structured_model = structured_model
        self.chat_model = chat_model
        self.final_prompt=""

    def process(self, input_text: str):
        # Step 1: Semantic search (side-effect or preloading)
        yield "Searching in document..."
        answer = self.search(input_text)
      
        # Step 2: Tagging
        yield "Making Tag prompts..."
        tag_prompt = self.tagger.invoke({"input": input_text})
        
        # Step 3: Structured classification
        yield "Structured Classification..."
        for key, value in self.structured_model.invoke(tag_prompt):
            yield f"{key}: {value}"

        # Step 4: Final prompt and chat
        yield "Finalizing Prompt..."
        self.final_prompt = format_prompt(input_text,answer)
       
        
            
    def invoke(self):
        for token in self.chat_model.stream(self.final_prompt):
            yield token.content
        
    