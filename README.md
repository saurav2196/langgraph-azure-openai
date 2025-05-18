# A Practical Guide to Building Multi-Agent Systems Using LangGraph and Azure OpenAI

In recent times, Large Language Models (LLMs) have opened incredible possibilities, but when tasks become complex, single-agent models often fall short. Enter LangGraph—a powerful framework built on LangChain designed explicitly for building sophisticated multi-agent, multi-step AI workflows. In this guide, I'll take you through practical aspects of LangGraph, explain why you'd want to use it with Azure OpenAI specifically, and provide hands-on examples with code snippets and detailed explanations.

## Why LangGraph with Azure OpenAI?

LangGraph extends LangChain capabilities, enabling:

* **Stateful Workflows:** Maintains state across different agents and steps.
* **Conditional Logic:** Supports branching logic based on intermediate results.
* **Agent Coordination:** Facilitates seamless collaboration among multiple specialized agents.

Integrating Azure OpenAI enhances LangGraph by leveraging Microsoft's enterprise-grade AI capabilities, including scalability, security, and seamless integration into existing Azure cloud environments.

## Setting Up LangGraph and Azure OpenAI

### Installation

```bash
pip install langgraph langchain openai azure-ai-openai python-dotenv pandas
```

### Azure OpenAI Configuration via .env File

Create a `.env` file in your project's root directory and add the following:

```env
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_ENDPOINT=<your-endpoint>
AZURE_OPENAI_VERSION=<api-version>
AZURE_GPT4O_MODEL=<deployment-name>
```

Load environment variables in your Python script:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Building a Simple LangGraph Workflow with Azure OpenAI

Define a simple multi-agent workflow:

* Planner Agent: Parses the user query and prepares a simulation plan.
* Weather Agent: Executes a weather forecasting model using Azure OpenAI.
* Analyst Agent: Summarizes the simulation results.

### Step 1: Define Your Simulation State

```python
from typing import TypedDict, Dict

class SimulationState(TypedDict):
    simulation_plan: Dict[str, str]
    weather_forecast: Dict[str, str]
    summary: str
```

### Step 2: Implement Nodes Using Azure OpenAI

```python
import os
from langgraph.graph import StateGraph
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_VERSION'),
    azure_deployment=os.getenv('AZURE_GPT4O_MODEL'),
    temperature=0.3
)

def planner_node(state: SimulationState) -> SimulationState:
    plan = {"date": "2025-06-01", "location": "Scotland", "condition": "wind drop 10%"}
    state["simulation_plan"] = plan
    return state

def weather_agent_node(state: SimulationState) -> SimulationState:
    plan = state["simulation_plan"]
    user_prompt = f"Forecast weather on {plan['date']} for {plan['location']} considering {plan['condition']}."

    response = llm.invoke([
        {"role": "system", "content": "Provide a weather forecast based on the given conditions."},
        {"role": "user", "content": user_prompt}
    ])

    state["weather_forecast"] = {"forecast": response.content.strip()}
    return state

def analyst_agent_node(state: SimulationState) -> SimulationState:
    plan = state["simulation_plan"]
    forecast = state["weather_forecast"]["forecast"]
    state["summary"] = f"Simulation on {plan['date']} for {plan['location']}: {forecast}"
    return state
```

### Step 3: Create and Connect Nodes

```python
builder = StateGraph(SimulationState)
builder.add_node("planner", planner_node)
builder.add_node("weather_agent", weather_agent_node)
builder.add_node("analyst", analyst_agent_node)

builder.set_entry_point("planner")
builder.add_edge("planner", "weather_agent")  # Planner creates the plan and passes it to Weather Agent
builder.add_edge("weather_agent", "analyst")  # Weather Agent forecasts and forwards results to Analyst Agent

# Graph Flow Representation
# planner -> weather_agent -> analyst

graph = builder.compile()
```

### Step 4: Run Your Workflow

```python
initial_state: SimulationState = {
    "simulation_plan": {},
    "weather_forecast": {},
    "summary": ""
}

final_output = graph.invoke(initial_state)

print("\n--- Simulation Summary ---")
print(final_output["summary"])
```

### Expected Output

```
--- Simulation Summary ---
--- Simulation Summary ---
Simulation on 2025-06-01 for Scotland: Predicting the weather for a specific date in the future, such as June 1, 2025, involves a lot of uncertainties, especially given the dynamic nature of weather patterns. However, I can provide a general forecast based on typical weather conditions for Scotland during early June and the impact of a 10% wind drop.

In early June, Scotland usually experiences mild temperatures, with averages ranging from 10°C to 17°C (50°F to 63°F). The weather can be quite variable, with a mix of sunny spells and scattered showers. Given the wind drop of 10%, you might expect slightly calmer conditions than usual, which could lead to less cloud movement and potentially more stable weather patterns.

Here's a speculative forecast for June 1, 2025, in Scotland:

- **Temperature:** Expect mild temperatures, likely between 12°C and 18°C (54°F to 64°F).
- **Precipitation:** There may be occasional light showers, but the reduced wind could mean less frequent rain.
- **Wind:** Winds will be lighter than usual, with speeds reduced by approximately 10%, leading to calmer conditions.
- **Sunshine:** With reduced wind, there might be longer sunny periods, especially in the afternoon.
- **Overall:** A mix of clouds and sunshine with a chance of light rain, but generally pleasant conditions for outdoor activities.

Please note that this forecast is speculative and based on typical patterns for the region during this time of year. For precise weather predictions, closer to the date, it's best to consult local meteorological services.
```

## Best Practices

* **Clearly define state objects:** Maintain clarity and readability.
* **Modularize nodes:** Each agent or step should independently handle a single responsibility.
* **Leverage Azure's reliability:** Ensure scalability and security in enterprise applications.

## What's Next?

In my next blog, I'll explore integrating LangSmith for detailed observability, monitoring, and debugging of LangGraph workflows, further enhancing your multi-agent system development experience.

---
