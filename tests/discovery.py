import os
import unittest

from langchain_gigachat import GigaChatEmbeddings

from spade_llm.builders import MessageBuilder
from spade_llm.consts import Templates
from spade_llm.discovery import AgentDescription, AgentTask, DirectoryFacilitatorAgent, AgentSearchRequest, \
    AgentSearchResponse
from tests.base import SpadeTestCase

DF_ADDRESS = "df@localhost"

math = AgentDescription(
    description = "This is a mathematical expert can perform computations, even complex ones, "
                  "and solve mathematical problems",
    id = "math@localhost",
    domain = "math",
    tier = 0,
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
    domain = "search",
    tier = 0,
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

travel = AgentDescription(
    description = "Assistant for the trip planning. Can help with tickets and hotels.",
    id = "travel@localhost",
    domain = "travel",
    tier = 0,
    tasks = [
        AgentTask(
            description = "Booking airplane tickets",
            examples = [
                "Book a ticket from London to Paris",
                "Get a flight to New-York.",
                "Get me to Moscow"
            ]),
        AgentTask(
            description = "Book hotels",
            examples = [
                "Get a hotel in Washington D.C. for friday",
                "Need a stop in Saint-Petersburg these weekends.",
                "Find accommodation in London for may."
            ])
    ]
)

class MyTestCase(SpadeTestCase):

    @classmethod
    def setUpClass(cls):
        SpadeTestCase.setUpClass()
        df = DirectoryFacilitatorAgent(
            DF_ADDRESS,
            "pwd",
              embeddings=GigaChatEmbeddings(
                  credentials=os.environ['GIGA_CRED'],
                  verify_ssl_certs=False,
              ))
        SpadeTestCase.startAgent(df)

        SpadeTestCase.sendAndReceive(
            MessageBuilder.inform().to_agent(df).with_content(math),
            Templates.ACKNOWLEDGE())
        SpadeTestCase.sendAndReceive(
            MessageBuilder.inform().to_agent(df).with_content(search),
            Templates.ACKNOWLEDGE())
        SpadeTestCase.sendAndReceive(
            MessageBuilder.inform().to_agent(df).with_content(travel),
            Templates.ACKNOWLEDGE())

    def search_foragent(self, query:str) -> AgentSearchResponse:
        msg = SpadeTestCase.sendAndReceive(
            MessageBuilder.request().to_agent(DF_ADDRESS)
                .with_content(AgentSearchRequest(task = query)),
            Templates.INFORM())
        self.assertIsNotNone(msg,"Received no search result")
        return AgentSearchResponse.model_validate_json(msg.body)

    def assertAgent(self, result: AgentSearchResponse, id: str):
        self.assertListEqual([id], [x.id for x in result.agents])

    def test_find_math(self):
        result = self.search_foragent(
            "Solve a quadratic equation 2x^2 + x - 5 = 0")
        self.assertAgent(result, math.id)

    def test_find_search(self):
        result =self.search_foragent(
            "When and who discovered antarctic continent?")
        self.assertAgent(result, search.id)

    def test_find_travel(self):
        result =self.search_foragent(
            "I would like to visit Belgium in may")
        self.assertAgent(result, travel.id)

if __name__ == '__main__':
    unittest.main()
