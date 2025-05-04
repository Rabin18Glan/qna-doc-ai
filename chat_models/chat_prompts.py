from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def format_prompt(user_message,answer):
    prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a best teacher in the world and Explaining everything in exactly what it is and where it is used"),
    ("human", "{msgs}"),
    ("system", "{answer}")
   ])

    prompt = prompt_template.invoke({"msgs":user_message,"answer": answer})
    
    return prompt