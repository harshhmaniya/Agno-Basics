from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.crawl4ai import Crawl4aiTools

agent = Agent(
    model=Ollama("llama3.2"),
    tools=[Crawl4aiTools(
        max_length=None
    )],
    show_tool_calls=True
)
agent.print_response("Tell me about https://github.com/agno-agi/agno")