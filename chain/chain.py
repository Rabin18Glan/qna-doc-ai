from langchain_core.runnables import Runnable, RunnableMap
from chat_models.chat_prompts import format_prompt
import streamlit as st
class Chain(Runnable):
    def __init__(self, search, tagger, structured_model, chat_model):
        self.search = search
        self.tagger = tagger
        self.structured_model = structured_model
        self.chat_model = chat_model

    def invoke(self, input_text: str):
        # Step 1: Semantic search (side-effect or preloading)
        answer = self.search(input_text)
        yield answer
        # Step 2: Tagging
        tag_prompt = self.tagger.invoke({"input": input_text})
        yield tag_prompt
        # Step 3: Structured classification
        structured_output = self.structured_model.invoke(tag_prompt)
        yield structured_output.model_dump()
        # Step 4: Final prompt and chat
        final_prompt = format_prompt(input_text,answer)
        yield final_prompt
        for token in self.chat_model.stream(final_prompt):
            yield token.content

        yield "Chat model response complete."