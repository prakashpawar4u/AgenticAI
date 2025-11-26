from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain.chat_models import ChatOpenAI

# -------------------------
# 1️⃣ Define state schema
# -------------------------
class State(TypedDict):
    topic: str
    research: str
    draft: str

# -------------------------
# 2️⃣ Create the LangGraph
# -------------------------
graph = StateGraph(State)

# -------------------------
# 3️⃣ Initialize OpenAI Chat Model
# -------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -------------------------
# 4️⃣ Define nodes (agents)
# -------------------------
def researcher(state: State):
    prompt = f"Research the topic: {state['topic']} in 3 sentences."
    response = llm([{"role": "user", "content": prompt}])
    research_text = response.content
    return {"research": research_text}

def writer(state: State):
    prompt = f"Write a concise article draft based on this research:\n{state['research']}"
    response = llm([{"role": "user", "content": prompt}])
    draft_text = response.content
    return {"draft": draft_text}

# -------------------------
# 5️⃣ Add nodes to the graph
# -------------------------
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)

# ----------------------
