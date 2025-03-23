import unittest

from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tests.base import ModelTestCase

class TestStructuredOutput(ModelTestCase):

    # Pydantic
    class Joke(BaseModel):
        """Joke to tell user."""

        setup: str = Field(description="The setup of the joke")
        punchline: str = Field(description="The punchline to the joke")
        rating: Optional[int] = Field(
            default=None, description="How funny the joke is, from 1 to 10"
        )

    def test_with_structured_output(self):
        structured_llm = self.lite.with_structured_output(self.Joke)

        joke = structured_llm.invoke("Tell me a joke about cats")
        print("From structured output: " + str(joke))
        self.assertIsInstance(joke, self.Joke)

    def test_with_output_parser(self):
        parser = PydanticOutputParser(pydantic_object=self.Joke)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the user query. Wrap the output in `json` tags\n{format_instructions}",
                ),
                ("human", "{query}"),
            ]
        ).partial(format_instructions=parser.get_format_instructions())
        chain = prompt | self.lite | parser

        joke = chain.invoke({"query" : "Tell me a joke about cats"})
        print("From output parser: " + str(joke))
        self.assertIsInstance(joke, self.Joke)


if __name__ == '__main__':
    unittest.main()
