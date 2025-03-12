from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv
load_dotenv()

profile_agent = Agent(
    name="Linkedin profile finder",
    model=Groq(id="llama-3.3-70b-versatile"),
    description="You are an linkedin profile finder agent that finds linkedin profiles based on a given query",
    instructions=["First search the name in web in linkedin search",
                  "If user gives additional information like location, company, etc. then use that to refine the search",
                  "If user gives a linkedin profile url, then use that to find the profile",
                  "Return the link of the first profile of the search results"],
    tools=[DuckDuckGoTools()],
    markdown=True,
    show_tool_calls=True,
    stream=True
)

profile_agent.print_response("Find linkedin profile of Harsh Maniya", stream=True)
