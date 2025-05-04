from pydantic import BaseModel, Field


class Classification(BaseModel):
     sentiment: str = Field(description="The sentiment of the text")
     aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10"
      )
     language: str = Field(description="The language the text is written in")
     question_difficulty: str = Field(description="How difficult the question is")


