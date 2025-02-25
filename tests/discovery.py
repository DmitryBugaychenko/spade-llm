import os
import unittest
from typing import Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_gigachat import GigaChatEmbeddings
from pydantic import BaseModel, Field
from itertools import chain

from tests.base import ModelTestCase


# Pydantic
class AgentTask(BaseModel):
    """Represents a single task agent is capable for."""

    description: str = Field(description="Description of a task")
    examples: Optional[list[str]] = Field(description="Examples with a task query")

    def to_string(self) -> str:
        return self.description + "\n" + "\n".join(self.examples)

class AgentDescription(BaseModel):
    """Represents a description of a certain agent with its functions"""

    description: str = Field(description="Description of the agent role and abilities")
    tasks: Optional[list[AgentTask]] = Field(description="Tasks agent is capable for", default = [])
    domain: Optional[str] = Field(description="Domain agent is expert at", default=None)
    tier: int = Field(default = 0, description="Tier the agent belongs to. During discovery process agents can "
                                               "lookup only agents in the tiers below and in the same domain.")
    id: str = Field(description="Unique identifier of an agent")

    def to_documents(self) -> list[Document]:
        yield Document(
            page_content=self.description,
            id = self.id,
            metadata = {"domain" : self.domain, "tier": self.tier}
        )

        for task in self.tasks:
            yield Document(
                page_content=self.description + "\n" + task.to_string(),
                id = self.id + "/" + task.description,
                metadata = {"domain" : self.domain, "tier": self.tier}
            )


math = AgentDescription(
    description = "This is a mathematical expert can perform computations, even complex ones, "
                  "and solve mathematical problems",
    id = "math@localhost",
    tasks = [
        AgentTask(
            description = "Equation Solving",
            examples = [
                "Solve polynomial equations.",
                "Solve systems of linear equations.",
                "Solve differential equations."
            ]),
        AgentTask(
            description = "Integration and Differentiation",
            examples = [
                "Evaluate definite integrals.",
                "Perform indefinite integration.",
                "Differentiate expressions."
            ]),
        AgentTask(
            description = "Linear Algebra operations",
            examples = [
                "Multiply two matrices.",
                "Find the eigenvalues and eigenvectors of matrix",
                "Find the cross product of two vectors"
            ]),
        AgentTask(
            description = "Complex Numbers and Polynomials",
            examples = [
                "Manipulate complex numbers.",
                "Factor polynomial completely over the complex numbers",
                "Determine if the polynomial has any rational root"
            ]),
        AgentTask(
            description = "Probability and Statistics",
            examples = [
                "What is the probability of rolling a sum of 7 with two standard dice?",
                "Find the mean and variance of a binomial distribution with parameters n=10 and p=0.3.",
                "Test whether the average height of students in a class is significantly different from 170 cm, given sample data"
            ]),
    ]
)

search = AgentDescription(
    description = "This is an information retrieval expert capable of finding information in the Internet and answer factual questions",
    id = "search@localhost",
    tasks = [
        AgentTask(
            description = "Answering Factual Questions",
            examples = [
                "When did World War II end?",
                "Explain the theory of relativity in layman's terms.",
                "What were the key outcomes of the latest climate summit?"
            ]),
        AgentTask(
            description = "Finding Specific Information Online",
            examples = [
                "Find the phone number of the Russian embassy in Washington D.C.",
                "Find internship programs in renewable energy companies in Europe.",
                "Find free online courses on machine learning."
            ]),
        AgentTask(
            description = "Customized Research Tasks",
            examples = [
                "Compile a list of books recommended for beginners interested in astronomy.",
                "Gather financial data for analyzing stock market performance over the past decade.",
                "Prepare a reading list on sustainable agriculture practices."
            ])
    ]
)

class MyTestCase(ModelTestCase):

    embeddings = GigaChatEmbeddings(
        credentials=os.environ['GIGA_CRED'],
        verify_ssl_certs=False,
    )

    vector_store = InMemoryVectorStore(embeddings)
    threshold = 0.76

    travel = Document(
        page_content="Assistant for the trip planning. Can help with tickets and hotels. Example "
                     "tasks include:"
                     "- Booking an airplane ticket"
                     "- Find an appropriate hotel"
                     "- Check visa requirement"
                     "- Plan a trip in a foreign country",
        id = "travel@localhost",
        metadata = {"domain":"travel", "tier":0}
    )

    vector_store.add_documents(
        documents= list(chain(math.to_documents(), search.to_documents(), [travel]))
    )


    def filter_results(self, result: list[(Document,float)]) -> list[(Document,float)]:
        documents = set()
        for doc, score in result:
            short_id = doc.id.split("/", 1)[0]
            if short_id not in documents:
                documents.add(short_id)
                yield (doc.model_copy(update={"id":short_id}), score)

    def test_find_math(self):
        result = self.search_foragent("Solve a quadratic equation 2x^2 + x - 5 = 0")
        self.assertAgent(result, math.id)

    def search_foragent(self, query:str):
        result = list(self.filter_results(self.vector_store.similarity_search_with_score(query)))
        print("For query {0}".format(query))
        for doc, score in result:
            print("Score: {0}, Doc {1}".format(score, doc.id))
        return result

    def test_find_search(self):
        result =self.search_foragent(
            "When and who discovered antarctic continent?")
        self.assertAgent(result, search.id)

    def test_find_travel(self):
        result =self.search_foragent(
            "I would like to visit Belgium in may")
        self.assertAgent(result, self.travel.id)


    def assertAgent(self, result, id: str):
        self.assertEqual(id, result[0][0].id)
        self.assertGreaterEqual(result[0][1], self.threshold)
        for doc, score in result[1:]:
            self.assertLess(score, self.threshold, "To high score for " + doc.id)


if __name__ == '__main__':
    unittest.main()
