from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from crewai import LLM
from flask import Flask, request, render_template

load_dotenv()

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.3
)

# ------------------ AGENTS ------------------

web_agent = Agent(
    name="Web Agent",
    role="Research Analyst",
    goal="Find factual information about {topic}",
    backstory="Expert at searching verified information.",
    tools=[SerperDevTool()],
    llm=llm,
    verbose=True,
)

writer_agent = Agent(
    name="Writer",
    role="Creative Writer",
    goal="Write summaries based on research",
    backstory="Skilled at simplifying complex topics.",
    llm=llm,
    verbose=True,
)

coder_agent = Agent(
    name="Code Agent",
    role="Python Developer",
    goal="Write clean runnable Python code when asked.",
    backstory="Outputs ONLY code, no explanation.",
    llm=llm,
    verbose=True,
    instructions="Return ONLY code inside ```python ... ```"
)

# ------------------ TASKS ------------------

info_task = Task(
    description="Research and summarize about: {topic}",
    expected_output="A clear explanation.",
    agent=web_agent
)

write_task = Task(
    description="Write a user-friendly summary about: {topic}",
    expected_output="100-word summary.",
    agent=writer_agent,
    context=[info_task]
)

code_task = Task(
    description="Write Python code requested by user: {topic}",
    expected_output="A code block only",
    agent=coder_agent
)

# ------------------ FLASK ------------------

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form.get("question").lower()

    # keywords to detect coding request
    coding_keywords = ["code", "python", "program", "script"]

    is_code = any(k in user_input for k in coding_keywords)

    if is_code:
        print(">>> CODE MODE ACTIVATED")
        # Create a temporary crew ONLY for the code task
        code_crew = Crew(
            agents=[coder_agent],
            tasks=[code_task],
            verbose=True,
        )
        result = code_crew.kickoff(inputs={"topic": user_input})
    else:
        print(">>> NORMAL INFO MODE")
        info_crew = Crew(
            agents=[web_agent, writer_agent],
            tasks=[info_task, write_task],
            verbose=True
        )
        result = info_crew.kickoff(inputs={"topic": user_input})

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)
