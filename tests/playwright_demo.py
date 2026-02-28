from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
import asyncio
import os

from playwright.async_api import async_playwright

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_gigachat.chat_models import GigaChat
from posthog import debug


async def main():

    print("Starting Playwright browser...")
    browser = await async_playwright().start()
    print("Launching Chromium browser...")
    async_browser = await browser.chromium.launch(headless=True, timeout=60000, devtools=True)
    print("Initializing browser toolkit...")
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    print("Toolkit initialized")
    tools = toolkit.get_tools()

    # tools_by_name = {tool.name: tool for tool in tools}
    # navigate_tool = tools_by_name["navigate_browser"]
    # get_elements_tool = tools_by_name["get_elements"]
    #
    # print("Running navigate tool...")
    # content = await navigate_tool.arun(
    #     {"url": "https://huggingface.co/models"}
    # )
    # print(content)
    #
    # elements = await get_elements_tool.arun(
    #     {"selector": ".overview-card-wrapper", "attributes": ["innerText"]}
    # )
    #
    # print("Got elements: " + elements)

    print("Creating LLM...")
    # llm = ChatOpenAI(
    #     model="deepseek/deepseek-chat-v3.1:free",
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key="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",
    # )

    llm = GigaChat(
        credentials=os.environ['GIGA_CRED'],
        verify_ssl_certs=False
    )

    print("Creating agent...")
    agent_chain = create_react_agent(
        model=llm,
        tools=tools,
        prompt="You are an information extraction agent. Use the provided tools to find information in the internet. "
               "Your tools operate a browser session, first navigate the webpage requested. If navigation returns code 200 use"
               "other tools to parse the page content.",
        debug=True)

    print("Sending message to agent...")
    result = await agent_chain.ainvoke(
        {"messages": [("user", "Сколько лет на рынке и чем занимается компания Мера?  Её сайт https://www.skmera.ru")]}
    )
    print("Got response: " + result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())