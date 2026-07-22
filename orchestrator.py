import re
import traceback
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END

from agents import (
    run_trainer_agent,
    run_yogi_agent,
    run_dietitian_agent,
    run_general_chat_agent,
)
from database import get_user_profile_string, get_recent_history, save_chat_message

# Regex pattern matching for 0ms classification
TRAINER_PATTERNS = r"\b(workout|exercise|gym|lift|bench|squat|deadlift|biceps|triceps|chest|legs|cardio|hiit|sets|reps|pushup|pullup|training|workout plan)\b"
YOGI_PATTERNS = r"\b(yoga|stretch|stretching|mobility|flexibility|pose|asana|hamstring|hip opener|joint pain|lower back)\b"
DIETITIAN_PATTERNS = r"\b(diet|nutrition|calories|protein|macros|meal|recipe|eat|food|carbs|fat|calorie|supplement)\b"

# =========================================================
# 1. State Definition
# =========================================================
class WellnessState(TypedDict):
    user_id: str
    user_message: str
    user_profile: str
    recent_history: str
    required_agents: List[str]
    trainer_plan: str
    yogi_plan: str
    dietitian_plan: str
    assembled_plan: str
    final_output: str

# =========================================================
# 2. Graph Node Functions
# =========================================================
def initialize_context_node(state: WellnessState) -> Dict[str, Any]:
    profile = get_user_profile_string(state["user_id"])
    history = get_recent_history(state["user_id"], turns=4) or "None"
    return {
        "user_profile": profile,
        "recent_history": history,
        "trainer_plan": "",
        "yogi_plan": "",
        "dietitian_plan": "",
    }

def fast_regex_router_node(state: WellnessState) -> Dict[str, Any]:
    """0ms Router: Uses Regex matching instead of making a slow LLM API call."""
    msg = state["user_message"].lower()
    selected = []

    if re.search(TRAINER_PATTERNS, msg):
        selected.append("trainer")
    if re.search(YOGI_PATTERNS, msg):
        selected.append("yogi")
    if re.search(DIETITIAN_PATTERNS, msg):
        selected.append("dietitian")

    # If no specialist keywords hit, default straight to General LLM
    if not selected:
        selected = ["general"]

    return {"required_agents": selected}

def trainer_node(state: WellnessState) -> Dict[str, Any]:
    plan = run_trainer_agent(state["user_profile"], state["user_message"])
    return {"trainer_plan": plan}

def yogi_node(state: WellnessState) -> Dict[str, Any]:
    plan = run_yogi_agent(state["user_profile"], state["user_message"], state.get("trainer_plan", ""))
    return {"yogi_plan": plan}

def dietitian_node(state: WellnessState) -> Dict[str, Any]:
    plan = run_dietitian_agent(state["user_profile"], state["user_message"], state.get("trainer_plan", ""))
    return {"dietitian_plan": plan}

def general_chat_node(state: WellnessState) -> Dict[str, Any]:
    output = run_general_chat_agent(
        user_message=state["user_message"], 
        recent_history=state["recent_history"],
        user_profile=state["user_profile"]
    )
    return {"final_output": output}

def compile_plan_node(state: WellnessState) -> Dict[str, Any]:
    sections = []
    if state.get("trainer_plan"):
        sections.append(f"### 🏋️ Workout\n{state['trainer_plan']}")
    if state.get("yogi_plan"):
        sections.append(f"### 🧘 Yoga\n{state['yogi_plan']}")
    if state.get("dietitian_plan"):
        sections.append(f"### 🥗 Nutrition\n{state['dietitian_plan']}")

    compiled = "\n\n".join(sections) if sections else state.get("final_output", "")
    return {"assembled_plan": compiled, "final_output": compiled}

# =========================================================
# 3. Router Conditionals
# =========================================================
def route_specialists(state: WellnessState) -> List[str]:
    agents = state.get("required_agents", [])
    if "general" in agents:
        return ["general_chat_node"]
    
    mapping = {
        "trainer": "trainer_node",
        "yogi": "yogi_node",
        "dietitian": "dietitian_node"
    }
    return [mapping[a] for a in agents if a in mapping]

# =========================================================
# 4. Construct State Graph
# =========================================================
workflow = StateGraph(WellnessState)

workflow.add_node("initialize_context_node", initialize_context_node)
workflow.add_node("fast_regex_router_node", fast_regex_router_node)
workflow.add_node("trainer_node", trainer_node)
workflow.add_node("yogi_node", yogi_node)
workflow.add_node("dietitian_node", dietitian_node)
workflow.add_node("general_chat_node", general_chat_node)
workflow.add_node("compile_plan_node", compile_plan_node)

workflow.add_edge(START, "initialize_context_node")
workflow.add_edge("initialize_context_node", "fast_regex_router_node")

workflow.add_conditional_edges(
    "fast_regex_router_node",
    route_specialists,
    {
        "trainer_node": "trainer_node",
        "yogi_node": "yogi_node",
        "dietitian_node": "dietitian_node",
        "general_chat_node": "general_chat_node"
    }
)

workflow.add_edge("trainer_node", "compile_plan_node")
workflow.add_edge("yogi_node", "compile_plan_node")
workflow.add_edge("dietitian_node", "compile_plan_node")

workflow.add_edge("compile_plan_node", END)
workflow.add_edge("general_chat_node", END)

wellness_orchestrator = workflow.compile()

# =========================================================
# 5. Execution Interface
# =========================================================
def execute_wellness_orchestration(user_id: str, user_message: str) -> str:
    cleaned_message = user_message.strip()
    if not cleaned_message:
        return "How can I help you today?"

    try:
        initial_inputs: WellnessState = {
            "user_id": user_id,
            "user_message": cleaned_message,
            "user_profile": "",
            "recent_history": "",
            "required_agents": [],
            "trainer_plan": "",
            "yogi_plan": "",
            "dietitian_plan": "",
            "assembled_plan": "",
            "final_output": "",
        }

        final_state = wellness_orchestrator.invoke(initial_inputs)
        final_output = final_state.get("final_output", "How can I help?")

        save_chat_message(user_id, "user", cleaned_message)
        save_chat_message(user_id, "assistant", final_output)

        return final_output

    except Exception as e:
        traceback.print_exc()
        raise Exception(f"Orchestration failure: {str(e)}")
