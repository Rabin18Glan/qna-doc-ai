from chat_models.chat_models import model
from classification.classification import Classification

structured_llm = model.with_structured_output(Classification)