import logging
import os
import unittest

from langchain_gigachat import GigaChatEmbeddings

from spade_llm import consts
from spade_llm.agents.testing import DummyAgent, ExecuteInContext
from spade_llm.core.api import AgentContext
from spade_llm.patterns.discovery import AgentDescription, AgentTask, DirectoryFacilitatorAgent, AgentSearchRequest, \
    AgentSearchResponse, DirectoryFacilitatorAgentConf, RegisterAgentBehaviour
from tests.test_utils import TestPlatform, AgentEntry

logging.basicConfig(level=logging.DEBUG)
DF_ADDRESS = "df"

math = AgentDescription(
    description = "This is a mathematical expert can perform computations, even complex ones, "
                  "and solve mathematical problems",
    id = "math",
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
    id = "search",
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
    id = "travel",
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

class DiscoveryTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.dummy_agent = DummyAgent("dummy")

        self.df_agent = DirectoryFacilitatorAgent(agent_type=DF_ADDRESS)

        embeddings = GigaChatEmbeddings(credentials=os.environ['GIGA_CRED'], verify_ssl_certs=False)

        self.platform = TestPlatform(
            agents=[
                AgentEntry(
                    agent = self.df_agent,
                    configuration = DirectoryFacilitatorAgentConf(model="test")),
                AgentEntry(agent=self.dummy_agent),
            ],
            wait_for={self.dummy_agent.agent_type},
            embedding_models={"test" : embeddings}
        )

        await self.platform.start()
        
        class Register(ExecuteInContext):
            async def execute(self, context: AgentContext):
                await context.inform(DF_ADDRESS).with_content(math)
                await context.inform(DF_ADDRESS).with_content(search)
                await context.inform(DF_ADDRESS).with_content(travel)
        
        self.dummy_agent.as_agent(Register())
        
        # Wait for 3 ACKNOWLEDGE messages
        for _ in range(3):
            msg = self.dummy_agent.get_message(60)
            self.assertIsNotNone(msg, "Expected ACKNOWLEDGE message not received")
            self.assertEqual(msg.performative, consts.ACKNOWLEDGE)

    async def asyncTearDown(self):
        self.dummy_agent.stop()
        await self.platform.stop()

    def search_foragent(self, query: str) -> AgentSearchResponse:
        class Send(ExecuteInContext):
            async def execute(self, context: AgentContext):
                await context.request(DF_ADDRESS) \
                    .with_content(AgentSearchRequest(task=query))
        
        self.dummy_agent.as_agent(Send())
        response_message = self.dummy_agent.get_message()
        return AgentSearchResponse.model_validate_json(response_message.content)

    def assertAgent(self, result: AgentSearchResponse, id: str):
        self.assertListEqual([id], [x.id for x in result.agents])

    def test_find_math(self):
        result = self.search_foragent(
            "Solve a quadratic equation 2x^2 + x - 5 = 0")
        self.assertAgent(result, math.id)

    def test_find_search(self):
        result = self.search_foragent(
            "When and who discovered antarctic continent?")
        self.assertAgent(result, search.id)

    def test_find_travel(self):
        result = self.search_foragent(
            "I would like to visit Belgium in may")
        self.assertAgent(result, travel.id)


if __name__ == '__main__':
    unittest.main()
