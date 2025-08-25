from agno.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.tools.website import WebsiteTools
from agno.team import Team
import google.generativeai as genai
from flask import Flask, request, render_template 
from flask import jsonify,json

# Configure Gemini with your API key
genai.configure(api_key="AIzaSyDuv8gf8AbDxOWiPfUZlbjlCguKXYolQZk")

# Web search agent
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=genai.GenerativeModel('gemini-1.5-flash'),
    tools=[DuckDuckGo(),WebsiteTools()],
    instructions=["Always include sources","use the all the search results"],
    show_tool_calls=True,
    markdown=True,
)

# Finance agent
finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=genai.GenerativeModel('gemini-1.5-flash'),
    tools=[YFinanceTools(stock_price=True),WebsiteTools()],
    instructions=["Always include sources","use the all the search results","go search for other websites and give me the results"],
    show_tool_calls=True,
    markdown=True,
)

# Team of agents
agent_team = Team(
    mode="coordinate",  # or "collaborate" depending on behavior you want
    members=[web_agent, finance_agent],
    model=genai.GenerativeModel("gemini-1.5-flash"),
    success_criteria="A comprehensive financial news report with clear sections and data-driven insights.",
    instructions=["Always include sources", "Use tables to display data","use all the search results"],
    show_tool_calls=True,
    markdown=True,
)

# ✅ Proper streaming output
response = agent_team.model.generate_content(
    "  ",
    stream=True
)

for chunk in response:
    if chunk.candidates and chunk.candidates[0].content.parts:
        print(chunk.candidates[0].content.parts[0].text, end="", flush=True)



app = Flask(__name__)


@app.route("/")
def home():
    return render_template("agent.index.html")

@app.route("/predict", methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    user_query = features[0]
    # Generate a response using the agent team
    response = agent_team.model.generate_content(user_query)
    # Extract the text from the response
    output = ""
    for chunk in response:
        if chunk.candidates and chunk.candidates[0].content.parts:
            output += chunk.candidates[0].content.parts[0].text
    
    #print(output)
    
    return render_template("agent.index.html", prediction_text=f'{output}')
    

if __name__ == "__main__":
    app.run(debug=True)
