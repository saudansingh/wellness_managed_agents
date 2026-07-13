from typing import TypedDict, Union, Optional, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agents import (
    run_trainer_agent, 
    run_yogi_agent, 
    run_dietitian_agent, 
    run_safety_agent, 
    analytical_pro_model
)
from database import (
    get_user_profile_string, 
    get_last_week_number, 
    save_weekly_plan,
    save_chat_message,
    get_chat_history_messages
)

class WellnessState(TypedDict):
    user_id: str
    user_message: str                
    required_agents: List[str]       
    week_number: Optional[int]
    user_profile: Optional[str]
    chat_history_context: List[dict]  # Feeds past message blocks to the LLM
    workout_plan: Optional[str]
    yoga_plan: Optional[str]
    diet_plan: Optional[str]
    loop_counter: int
    safety_status: Optional[str]
    final_output: Optional[str]

def initialize_workflow_node(state: WellnessState) -> dict:
    user_id = state["user_id"]
    profile_str = get_user_profile_string(user_id)
    current_max_week = get_last_week_number(user_id)
    
    # Fix Amnesia: Dynamically pull the latest turns from the DB
    history = get_chat_history_messages(user_id, limit=10)
    
    return {
        "user_profile": profile_str,
        "week_number": current_max_week + 1,
        "chat_history_context": history,
        "workout_plan": state.get("workout_plan", ""),
        "yoga_plan": state.get("yoga_plan", ""),
        "diet_plan": state.get("diet_plan", ""),
        "loop_counter": state.get("loop_counter", 0) + 1,
        "safety_status": "",
        "final_output": ""
    }

def intent_analyzer_node(state: WellnessState) -> dict:
    """Uses past chat turns as context so follow-ups like 'well now' are clear."""
    message = state["user_message"]
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in state["chat_history_context"]])
    
    routing_prompt = f"""You are a precise routing AI for a managed wellness agent. 
    Analyze the current user request within the context of recent chat turns.
    
    Recent Context History:
    {history_str}

    Available specialists:
    - "trainer" (exercises, gym routines, workout strategies)
    - "yogi" (stretching, mobility, yoga poses)
    - "dietitian" (food, recipes, macros, diet plans)
    - "casual" (greetings, general chat, acknowledgments like 'it was good', 'well now', or out-of-domain queries)

    User Request: "{message}"
    Return ONLY a single comma-separated list of the keys needed. Do not write full sentences.
    Result:"""
    
    try:
        response = analytical_pro_model.invoke(routing_prompt).content.strip().lower()
        required = [w.strip() for w in response.replace('"', '').split(",") if w.strip() in ["trainer", "yogi", "dietitian", "casual"]]
    except Exception:
        required = ["casual"]
        
    if not required:
        required = ["casual"]
        
    return {"required_agents": required}

def casual_chat_node(state: WellnessState) -> dict:
    if "casual" not in state["required_agents"]:
        return {}
        
    # Fix Out-of-Bounds Hijacking: Implement strict professional system guardrails
    chat_prompt = f"""You are an elite, professional AI Wellness Assistant. 
    You manage client relationship loops, general check-ins, or casual remarks.

    CRITICAL DISCIPLINE GUARDRAIL: You are restricted entirely to personal health and wellness boundaries. 
    You possess zero knowledge about general trivia, current news events, celebrity culture, or politics. 
    If the user asks general knowledge questions (e.g., 'Who is the PM of New Zealand?'), you MUST explicitly and politely refuse to answer, stating you are uniquely optimized for wellness goals.

    User message: {state['user_message']}
    Response:"""
    
    response = analytical_pro_model.invoke(chat_prompt).content.strip()
    return {"final_output": response}

# ... [Keep trainer_node, yogi_node, dietitian_node, safety_audit_node, and handle_medical_refusal_node as they are] ...

def finalize_and_save_node(state: WellnessState) -> dict:
    complete_markdown_plan = ""
    if state.get("workout_plan"): complete_markdown_plan += f"{state['workout_plan']}\n\n"
    if state.get("yoga_plan"): complete_markdown_plan += f"{state['yoga_plan']}\n\n"
    if state.get("diet_plan"): complete_markdown_plan += f"{state['diet_plan']}\n\n"
    
    if not complete_markdown_plan.strip():
        complete_markdown_plan = "# Response\nProcessing complete."
        
    save_weekly_plan(
        user_id=state["user_id"],
        week_number=state["week_number"],
        workout=state.get("workout_plan"),
        yoga=state.get("yoga_plan"),
        diet=state.get("diet_plan")
    )
    
    return {"final_output": complete_markdown_plan.strip()}

# =========================================================
# Graph Compiler Setup (Wiring remains standard)
# =========================================================
workflow = StateGraph(WellnessState)
workflow.add_node("initialize", initialize_workflow_node)
workflow.add_node("intent_analyzer", intent_analyzer_node)
workflow.add_node("trainer", trainer_node)
workflow.add_node("yogi", yogi_node)
workflow.add_node("dietitian", dietitian_node)
workflow.add_node("safety_audit", safety_audit_node)
workflow.add_node("handle_medical_refusal", handle_medical_refusal_node)
workflow.add_node("finalize_and_save", finalize_and_save_node)
workflow.add_node("casual_chat", casual_chat_node)

workflow.add_edge(START, "initialize")
workflow.add_edge("initialize", "intent_analyzer")

def route_to_agents(state: WellnessState):
    if state.get("loop_counter", 0) > 3: return "handle_medical_refusal"
    required = state.get("required_agents", [])
    if "casual" in required: return ["casual"]
    return required

workflow.add_conditional_edges("intent_analyzer", route_to_agents, {
    "casual": "casual_chat", "trainer": "trainer", "yogi": "yogi", "dietitian": "dietitian", "handle_medical_refusal": "handle_medical_refusal"
})

workflow.add_edge("trainer", "safety_audit")
workflow.add_edge("yogi", "safety_audit")
workflow.add_edge("dietitian", "safety_audit")

def evaluate_safety_gate(state: WellnessState):
    if "CRITICAL REJECTION" in state.get("safety_status", "").upper(): return "handle_medical_refusal"
    return "finalize_and_save"

workflow.add_conditional_edges("safety_audit", evaluate_safety_gate, {
    "handle_medical_refusal": "handle_medical_refusal", "finalize_and_save": "finalize_and_save"
})

workflow.add_edge("handle_medical_refusal", END)
workflow.add_edge("finalize_and_save", END)
workflow.add_edge("casual_chat", END)

wellness_orchestrator = workflow.compile()

def execute_wellness_orchestration(user_id: str, user_message: str) -> str:
    """Executes the pipeline and commits history transactions to database tables."""
    # 1. Log incoming user query
    save_chat_message(user_id=user_id, role="user", content=user_message)
    
    initial_inputs = {
        "user_id": user_id, 
        "user_message": user_message, 
        "required_agents": [],
        "loop_counter": 0 
    }
    
    final_state = wellness_orchestrator.invoke(initial_inputs)
    bot_response = final_state["final_output"]
    
    # 2. Log resulting bot reaction string 
    save_chat_message(user_id=user_id, role="assistant", content=bot_response)
    return bot_response
