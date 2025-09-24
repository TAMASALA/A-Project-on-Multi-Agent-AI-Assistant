from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool,CodeInterpreterTool
from crewai import LLM
from crewai_tools import WebsiteSearchTool,RagTool
from phi.tools.yfinance import YFinanceTools
from flask import Flask,request,render_template



load_dotenv()
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.1
)

code_interpreter = CodeInterpreterTool()


web_agent = Agent(
    name="Web Agent",
    role="Get information from the website",
    goal="Research interesting facts about the topic: {topic}",
    llm=llm,
    tools=[SerperDevTool()],
    instructions=["Always include sources"],
    backstory="You are an expert at finding relevant and factual data.",
    show_tool_calls=True,
    markdown=True,
    verbose=True,
)
sport_agent = Agent(
    name="Sports Agent",
    role="Get sports data",
    goal="Research key sports events and statistics for the topic: {topic}",
    llm=llm,
    tools=[SerperDevTool()],
    instructions=["Use tables to display data"],
    backstory="You are skilled at extracting sports information from various sources.",
    verbose=True,
    show_tool_calls=True,
    markdown=True,
)



writer_agent = Agent(
    role="Creative Writer",
    goal="Write a short blog summary using the research",
    backstory="You are skilled at writing engaging summaries based on provided content.",
    llm=llm,
    verbose=True,
)

task1 = Task(
    description="Find 3-5 interesting and recent facts about {topic} as of year 2025.",
    expected_output="A bullet list of 3-5 facts",
    agent=web_agent,
)
task2 = Task(
    description="Write a 100-word blog post summary about {topic} using the facts from the research.",
    expected_output="A blog post summary",
    agent=writer_agent,
    context=[task1],
)

crew = Crew(
    agents=[web_agent,sport_agent,writer_agent],
    tasks=[task1, task2],
    verbose=True,
)

crew.kickoff(inputs={"topic":"who is the cm of Andhra pradesh?"})

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form.get("question")
    result = crew.kickoff(inputs={"topic": user_input})
    output = {"answer": str(result)}
    return render_template("index.html", prediction_text=output["answer"])

if __name__ == "__main__":
    app.run(debug=True)