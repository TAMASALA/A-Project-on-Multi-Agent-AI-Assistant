🤖 Multi-Agent CrewAI System with Flask

This project demonstrates how to build a multi-agent AI system using the CrewAI framework and integrate it into a Flask web application.
The system is designed to answer user queries by combining the strengths of multiple specialized agents (Web Agent, Sports Agent, and Writer Agent).

📌 Features

Multi-Agent Collaboration:

Web Agent → fetches recent factual information from the web

Sports Agent → collects sports-related stats and events

Writer Agent → generates engaging summaries/blog posts

Custom LLM Integration: Uses Gemini (Google) as the language model with a low temperature for factual consistency.

Tool Usage:

SerperDevTool for web search

CodeInterpreterTool for computations

Flask Web Interface for user interaction

End-to-End Flow: User enters a query → CrewAI agents collaborate → Flask app displays result.

🛠️ Tech Stack

Programming Language: Python

Frameworks/Libraries:

CrewAI
 – Multi-agent system

Flask – Web application framework

SerperDevTool – Search engine tool

Gemini (LLM) – Text generation

Other Tools: dotenv, HTML (Flask templates)
