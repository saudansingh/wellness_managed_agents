import traceback
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END

from agents import (
    run_trainer_agent,
    run_yogi_agent,
    run_dietitian_agent,
    run_safety_agent,
    run_general_chat_agent,
    analytical_pro_model,
)
from database import get_user_profile_string, get_recent_history, save_chat_message

# List of common casual greetings to bypass LLM classification latency
FAST_GREETINGS = {
    "hi", "hello", "hey", "good morning", "good evening", 
    "good afternoon", "yo", "sup", "howdy", "hi there", "hello there"
}

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
    safety_status: str
    final_output: str

# =========================================================
# 2. Graph Node Functions
# =========================================================
def initialize_context_node(state: WellnessState) -> Dict[str, Any]:
    profile = get_user_profile_string(state["user_id"])
    history = get_recent_history(state["user_id"], turns=6) or "No prior history"
    return {
        "user_profile": profile,
        "recent_history": history,
        "trainer_plan": "",
        "yogi_plan": "",
        "dietitian_plan": "",
    }

def intent_router_node(state: WellnessState) -> Dict[str, Any]:
    """
    Analyzes user intent. Includes FAST-PATH to bypass LLM routing for 
    simple greetings or ultra-short messages (saving ~2 seconds).
    """
    msg = state["user_message"].strip()
    msg_clean = msg.lower().strip("!.,?")

    # ⚡ FAST-PATH: Instant routing for greetings / short chat (0 ms latency cost)
    if msg_clean in FAST_GREETINGS or len(msg_clean.split()) <= 2:
        return {"required_agents": ["general"]}

    prompt = f"""Analyze the user's message and categorize its intent into one or more labels:
- 'trainer': Workout, exercises, lifting, physical activity, strength, or movement.
- 'yogi': Yoga, flexibility, mobility, stretching, or joint pain.
- 'dietitian': Nutrition, diet, meals, calories, macros, or recipes.
- 'general': Greetings, casual chat, non-wellness topics, coding, math, general questions.

User message: "{msg}"

Respond with ONLY a comma-separated list of labels (e.g., 'trainer' or 'general')."""

    response = analytical_pro_model.invoke(prompt)
    raw_content = str(response.content).strip().lower().replace("`", "").replace('"', "").replace("'", "")
    
    parsed_labels = [label.strip() for label in raw_content.split(",") if label.strip()]
    valid_specialists = {"trainer", "yogi", "dietitian"}
    selected_agents = [l for l in parsed_labels if l in valid_specialists]
    
    if not selected_agents:
        return {"required_agents": ["general"]}
        
    return {"required_agents": selected_agents}

def trainer_node(state: WellnessState) -> Dict[str, Any]:
    plan = run_trainer_agent(state["user_profile"], state["user_message"])
    return {"trainer_plan": plan}

def yogi_node(state: WellnessState) -> Dict[str, Any]:
    context_workout = state.get("trainer_plan", "")
    plan = run_yogi_agent(state["user_profile"], state["user_message"], context_workout)
    return {"yogi_plan": plan}

def dietitian_node(state: WellnessState) -> Dict[str, Any]:
    context_workout = state.get("trainer_plan", "")
    plan = run_dietitian_agent(state["user_profile"], state["user_message"], context_workout)
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
        sections.append(f"### 🏋️ Workout Protocol\n{state['trainer_plan']}")
    if state.get("yogi_plan"):
        sections.append(f"### 🧘 Yoga & Mobility\n{state['yogi_plan']}")
    if state.get("dietitian_plan"):
        sections.append(f"### 🥗 Nutrition & Fuel\n{state['dietitian_plan']}")

    compiled = "\n\n---\n\n".join(sections) if sections else state.get("final_output", "")
    return {"assembled_plan": compiled}

def safety_audit_node(state: WellnessState) -> Dict[str, Any]:
    plan = state.get("assembled_plan", "")
    if not plan:
        return {"safety_status": "COMPLIANCE PASSED", "final_output": state.get("final_output", "")}

    status = run_safety_agent(state["user_profile"], plan)
    if "CRITICAL REJECTION" in status:
        return {
            "safety_status": status,
            "final_output": f"⚠️ **Safety Advisory**: This generated plan was flagged by safety checks:\n\n{status}"
        }
    return {"safety_status": "COMPLIANCE PASSED", "final_output": plan}

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
workflow.add_node("intent_router_node", intent_router_node)
workflow.add_node("trainer_node", trainer_node)
workflow.add_node("yogi_node", yogi_node)
workflow.add_node("dietitian_node", dietitian_node)
workflow.add_node("general_chat_node", general_chat_node)
workflow.add_node("compile_plan_node", compile_plan_node)
workflow.add_node("safety_audit_node", safety_audit_node)

workflow.add_edge(START, "initialize_context_node")
workflow.add_edge("initialize_context_node", "intent_router_node")

workflow.add_conditional_edges(
    "intent_router_node",
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

workflow.add_edge("compile_plan_node", "safety_audit_node")
workflow.add_edge("safety_audit_node", END)
workflow.add_edge("general_chat_node", END)

wellness_orchestrator = workflow.compile()

# =========================================================
# 5. Public Execution Interface
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
            "safety_status": "",
            "final_output": "",
        }

        final_state = wellness_orchestrator.invoke(initial_inputs)
        final_output = final_state.get("final_output", "How can I assist you with your fitness or nutrition?")

        save_chat_message(user_id, "user", cleaned_message)
        save_chat_message(user_id, "assistant", final_output)

        return final_output

    except Exception as e:
        traceback.print_exc()
        raise Exception(f"Orchestration failure: {str(e)}")
