import traceback
from typing import TypedDict, List, Dict, Any, Union
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
    """Loads profile context and recent dialogue history for the user."""
    profile = get_user_profile_string(state["user_id"])
    history = get_recent_history(state["user_id"], turns=4) or "No prior history"
    return {
        "user_profile": profile,
        "recent_history": history,
        "trainer_plan": "",
        "yogi_plan": "",
        "dietitian_plan": "",
    }

def intent_router_node(state: WellnessState) -> Dict[str, Any]:
    """
    Analyzes user intent to determine whether to trigger specialist agents 
    or default to natural chit-chat.
    """
    msg = state["user_message"].strip()
    
    prompt = f"""Analyze the user's message and categorize its primary intent into exactly one of these labels:
- 'trainer': Workout, lifting, strength, cardio, physical exercise, exercise form, or movement goals.
- 'yogi': Yoga, flexibility, mobility, stretching, joint stiffness, or alignment.
- 'dietitian': Nutrition, meals, diet, calories, macros, recipes, weight gain, or weight loss.
- 'greeting': Simple small talk, casual greetings, or general non-fitness chatter.
- 'unclear': Off-topic questions, general non-wellness questions, or ambiguous requests.

User message: "{msg}"
Respond with ONLY the exact label string."""

    response = analytical_pro_model.invoke(prompt)
    label = response.content.strip().lower()

    if label in ["trainer", "yogi", "dietitian"]:
        return {"required_agents": [label]}
    elif label in ["greeting", "unclear"]:
        return {"required_agents": ["general"]}
    else:
        # Default fallback to general chat
        return {"required_agents": ["general"]}

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
    output = run_general_chat_agent(state["user_message"], state["recent_history"])
    return {"final_output": output}

def compile_plan_node(state: WellnessState) -> Dict[str, Any]:
    """Combines specialized outputs into a unified plan format."""
    sections = []
    if state.get("trainer_plan"):
        sections.append(f"### 🏋️ Workout Protocol\n{state['trainer_plan']}")
    if state.get("yogi_plan"):
        sections.append(f"### 🧘 Yoga & Mobility\n{state['yogi_plan']}")
    if state.get("dietitian_plan"):
        sections.append(f"### 🥗 Nutrition & Fuel\n{state['dietitian_plan']}")

    compiled = "\n\n---\n\n".join(sections) if sections else state.get("user_message", "")
    return {"assembled_plan": compiled}

def safety_audit_node(state: WellnessState) -> Dict[str, Any]:
    """Audits generated wellness strategies against safety guidelines."""
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

# Add Nodes
workflow.add_node("initialize_context_node", initialize_context_node)
workflow.add_node("intent_router_node", intent_router_node)
workflow.add_node("trainer_node", trainer_node)
workflow.add_node("yogi_node", yogi_node)
workflow.add_node("dietitian_node", dietitian_node)
workflow.add_node("general_chat_node", general_chat_node)
workflow.add_node("compile_plan_node", compile_plan_node)
workflow.add_node("safety_audit_node", safety_audit_node)

# Flow Connections
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

# Merge specialist nodes into compiler
workflow.add_edge("trainer_node", "compile_plan_node")
workflow.add_edge("yogi_node", "compile_plan_node")
workflow.add_edge("dietitian_node", "compile_plan_node")

# Audit compiled plans
workflow.add_edge("compile_plan_node", "safety_audit_node")
workflow.add_edge("safety_audit_node", END)

# General chat bypasses compilation/safety audit
workflow.add_edge("general_chat_node", END)

# Compile Graph Instance
wellness_orchestrator = workflow.compile()

# =========================================================
# 5. Public Execution Interface
# =========================================================
def execute_wellness_orchestration(user_id: str, user_message: str) -> str:
    """
    Main entry point called by FastAPI main.py to handle both specialized requests 
    and general conversations smoothly.
    """
    cleaned_message = user_message.strip()
    if not cleaned_message:
        return "I didn't receive a message. What would you like help with today?"

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
        final_output = final_state.get("final_output", "I'm here to help with your workout, yoga, or nutrition goals!")

        # Log conversation to database history
        save_chat_message(user_id, "user", cleaned_message)
        save_chat_message(user_id, "assistant", final_output)

        return final_output

    except Exception as e:
        traceback.print_exc()
        raise Exception(f"Orchestration failure during execution: {str(e)}")
