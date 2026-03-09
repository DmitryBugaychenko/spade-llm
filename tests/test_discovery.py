import logging
import unittest

from spade_llm.agents.dummy import DummyAgent, ExecuteInContext
from spade_llm.core.behaviors import MessageTemplate, ContextBehaviour
from spade_llm.core.testing import TestPlatform, AgentEntry
from spade_llm.patterns.discovery import AgentDescription, AgentTask, DirectoryFacilitatorAgent, AgentSearchRequest, \
    AgentSearchResponse, DirectoryFacilitatorAgentConf
from tests.base import ModelTestCase

logging.basicConfig(level=logging.INFO)
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



class DiscoveryTest(ModelTestCase):

    def test_discovery_service(self):
        dummy_agent = DummyAgent()

        df_agent = DirectoryFacilitatorAgent(agent_type=DF_ADDRESS)

        test = self

        class RegisterAgent(ExecuteInContext):
            def __init__(self, description: AgentDescription):
                super().__init__()
                self.description = description

            async def execute(self, beh: ContextBehaviour):
                with test.subTest("Test registration for " + self.description.id):
                    await beh.context.inform(DF_ADDRESS).with_content(self.description)
                    ack = await beh.receive(MessageTemplate.acknowledge(), 10)
                    test.assertIsNotNone(ack)

        class SearchFor(ExecuteInContext):
            def __init__(self, query: str, expected: str):
                self.expected = expected
                self.query = query
            async def execute(self, beh: ContextBehaviour):
                await beh.context.request(DF_ADDRESS) \
                    .with_content(AgentSearchRequest(task=self.query))
                with test.subTest("Test search for " + self.expected):
                    response = await beh.receive(MessageTemplate.inform(), 10)
                    parsed = AgentSearchResponse.model_validate_json(response.content)

                    test.assertEqual(len(parsed.agents), 1)
                    test.assertEqual(parsed.agents[0].id, self.expected)

        dummy_agent.as_agent(RegisterAgent(math))
        dummy_agent.as_agent(RegisterAgent(search))
        dummy_agent.as_agent(RegisterAgent(travel))
        dummy_agent.as_agent(SearchFor("Solve a quadratic equation 2x^2 + x - 5 = 0", math.id))
        dummy_agent.as_agent(SearchFor("When and who discovered antarctic continent?", search.id))
        dummy_agent.as_agent(SearchFor("I would like to visit Belgium in may", travel.id))

        TestPlatform.run_test(
            agents=[
                AgentEntry(
                    agent = df_agent,
                    configuration = DirectoryFacilitatorAgentConf(model="test")),
                AgentEntry(agent=dummy_agent),
                ],
            wait_for={dummy_agent.agent_type},
            embedding_models={"test" : self.embeddings}
        )


if __name__ == '__main__':
    unittest.main()
