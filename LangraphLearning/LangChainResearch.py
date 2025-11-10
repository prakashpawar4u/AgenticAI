from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from groq import Groq
from dotenv import load_dotenv
from langsmith import Client as LangSmithClient
import os
import json
import time

# -------------------------
# 1️⃣ Load API keys
# -------------------------
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))#LANGCHAIN_HANDLER_API_KEY
ls_client = LangSmithClient(api_key=os.environ.get("LANGCHAIN_HANDLER_API_KEY"))

model_name = "llama-3.1-8b-instant"

# -------------------------
# 2️⃣ Define state schema
# -------------------------
class State(TypedDict):
    topic: str
    research: str
    draft: str

# -------------------------
# 3️⃣ Create the graph
# -------------------------
graph = StateGraph(State)

# -------------------------
# 4️⃣ Logging helper (for langsmith ≤0.4.x)
# -------------------------
def log_to_langsmith(name: str, input_state: dict, output_state: dict):
    """Create and finish a run using the old LangSmith API."""
    try:
        run = ls_client.create_run(
            name=name,
            inputs=input_state,
            run_type="chain",
            project_name="LangGraph-Learning"
        )
        if run and getattr(run, "id", None):
            ls_client.update_run(
                run.id,
                outputs=output_state,
                status="completed",
                end_time=time.time()
            )
            print(f"✅ Logged and completed run for node '{name}'")
        else:
            print(f"⚠️ Run creation returned None for node '{name}' — check API key or project name.")
    except Exception as e:
        print(f"⚠️ LangSmith logging failed for node '{name}': {e}")

# -------------------------
# 5️⃣ Define nodes
# -------------------------
def researcher(state: State):
    prompt = f"Research the topic: {state['topic']} in 3 sentences."
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    research_text = response.choices[0].message.content
    output_state = {"research": research_text}
    log_to_langsmith("researcher", state, output_state)
    return output_state

def writer(state: State):
    prompt = f"Write a concise article draft based on this research:\n{state['research']}"
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    draft_text = response.choices[0].message.content
    output_state = {"draft": draft_text}
    log_to_langsmith("writer", state, output_state)
    return output_state

# -------------------------
# 6️⃣ Connect the graph
# -------------------------
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_edge(START, "researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", END)

# -------------------------
# 7️⃣ Run the app
# -------------------------
app = graph.compile()
initial_state = {"topic": "Impact of renewable energy in India", "research": "", "draft": ""}
result = app.invoke(initial_state)

# -------------------------
# 8️⃣ Output
# -------------------------
print("\n=== FINAL STATE ===")
print(json.dumps(result, indent=4, ensure_ascii=False))
