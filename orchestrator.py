from typing import TypedDict, Union, Optional, List, Any
from langgraph.graph import StateGraph, START, END

# Import the specialist agent functions and models cleanly
from agents import (
    run_trainer_agent, 
    run_yogi_agent, 
    run_dietitian_agent, 
    run_safety_agent, 
    specialist_flash_model, 
    analytical_pro_model
)
from database import get_user_profile_string, get_last_week_number, save_weekly_plan

# =========================================================
# 1. Define the Shared Graph State
# =========================================================
class WellnessState(TypedDict):
    user_id: str
    user_message: str                
    required_agents: List[str]       
    week_number: Optional[int]
    user_profile: Optional[str]
    
    workout_plan: Optional[str]
    yoga_plan: Optional[str]
    diet_plan: Optional[str]
    loop_counter: int
    
    safety_status: Optional[str]
    final_output: Optional[str]

# =========================================================
# 2. Define the Graph Nodes
# =========================================================

def initialize_workflow_node(state: WellnessState) -> dict:
    user_id = state["user_id"]
    profile_str = get_user_profile_string(user_id)
    current_max_week = get_last_week_number(user_id)
    next_week = current_max_week + 1
    
    # Safely track current loop iterations
    current_loops = state.get("loop_counter", 0)
    
    return {
        "user_profile": profile_str,
        "week_number": next_week,
        "workout_plan": state.get("workout_plan", ""),
        "yoga_plan": state.get("yoga_plan", ""),
        "diet_plan": state.get("diet_plan", ""),
        "loop_counter": current_loops + 1, # Track loop cycles accurately
        "safety_status": "",
        "final_output": ""
    }

def intent_analyzer_node(state: WellnessState) -> dict:
    """Intelligently maps intents and toggles ONLY the required agent(s)."""
    message = state["user_message"]
    
    routing_prompt = f"""You are a strict, precise routing AI. Analyze the user request and determine WHICH specialist is needed.
    Available specialists:
    - "trainer" (exercises, gym routines, cardio, weight lifting, muscle building)
    - "yogi" (stretching, mobility, yoga, joint pain relief, flexibility)
    - "dietitian" (food, recipes, macros, calories, diet, weight loss/gain nutrition)

    CRITICAL RULE: Return ONLY a comma-separated list of the exact words needed. Do NOT write sentences. Do not include quotes.
    Example 1: dietitian
    Example 2: trainer, dietitian
    Example 3: yogi
    
    User Request: "{message}"
    Result:"""
    
    try:
        response = analytical_pro_model.invoke(routing_prompt).content.strip().lower()
        clean_string = response.replace('"', '').replace("'", "").replace(".", "")
        required = [word.strip() for word in clean_string.split(",") if word.strip() in ["trainer", "yogi", "dietitian"]]
    except Exception:
        required = []
        
    if not required:
        required = ["trainer"]
        
    print(f"🧠 [AI Router] User intent detected. Waking up ONLY: {required}")
    return {"required_agents": required}

def trainer_node(state: WellnessState) -> dict:
    if "trainer" not in state["required_agents"]:
        return {"workout_plan": ""}
        
    print("🏋️ [Agent] Trainer activated.")
    workout = run_trainer_agent(state["user_profile"], state["user_message"])
    return {"workout_plan": workout}

def yogi_node(state: WellnessState) -> dict:
    if "yogi" not in state["required_agents"]:
        return {"yoga_plan": ""}
        
    print("🧘 [Agent] Yogi activated.")
    workout_context = state.get("workout_plan") or "No workout context provided."
    yoga = run_yogi_agent(state["user_profile"], state["user_message"], workout_context)
    return {"yoga_plan": yoga}

def dietitian_node(state: WellnessState) -> dict:
    if "dietitian" not in state["required_agents"]:
        return {"diet_plan": ""}
        
    print("🥗 [Agent] Dietitian activated.")
    workout_context = state.get("workout_plan") or "No workout planned."
    yoga_context = state.get("yoga_plan") or "No yoga planned."
    workload = f"Workout splits:\n{workout_context}\n\nYoga recovery:\n{yoga_context}"
    
    diet = run_dietitian_agent(state["user_profile"], state["user_message"], workload)
    return {"diet_plan": diet}

def safety_audit_node(state: WellnessState) -> dict:
    print("🛡️ [Agent] Safety Auditor evaluating plan safety parameters...")
    
    combined_plan = f"--- USER REQUEST RESPONSE ---\n\n"
    if state.get("workout_plan"):
        combined_plan += f"{state['workout_plan']}\n\n"
    if state.get("yoga_plan"):
        combined_plan += f"{state['yoga_plan']}\n\n"
    if state.get("diet_plan"):
        combined_plan += f"{state['diet_plan']}\n\n"
        
    audit_result = run_safety_agent(state["user_profile"], combined_plan)
    print(f"🔍 [Safety Auditor Output Raw]: '{audit_result}'")
    
    return {"safety_status": audit_result.strip()}

def handle_medical_refusal_node(state: WellnessState) -> dict:
    disclaimer = (
        "### ⚠️ Strict Medical Notice\n"
        "Based on your profile, I cannot safely answer this strategy request without increasing health risks.\n\n"
        "Please consult a licensed physician before starting any training or diet adjustments."
    )
    return {"final_output": disclaimer}

def finalize_and_save_node(state: WellnessState) -> dict:
    complete_markdown_plan = ""
    
    if state.get("workout_plan"):
        complete_markdown_plan += f"{state['workout_plan']}\n\n"
    if state.get("yoga_plan"):
        complete_markdown_plan += f"{state['yoga_plan']}\n\n"
    if state.get("diet_plan"):
        complete_markdown_plan += f"{state['diet_plan']}\n\n"
        
    if not complete_markdown_plan.strip():
        complete_markdown_plan = "# Response\nI could not find a specific answer module for your request."
    
    save_weekly_plan(
        user_id=state.get("user_id", "default"),
        week_number=state.get("week_number", 1),
        workout=state.get("workout_plan"),
        yoga=state.get("yoga_plan"),
        diet=state.get("diet_plan")
    )
    
    return {"final_output": complete_markdown_plan.strip()}

# =========================================================
# 3. Conditional Routing Logic Functions
# =========================================================

def route_to_agents(state: WellnessState) -> Union[str, List[str]]:
    """Determines which agent paths to follow out of the intent_analyzer node."""
    # 1. Catch infinite recursive execution loops immediately
    if state.get("loop_counter", 0) > 3:
        print("🚨 [Circuit Breaker] Loop counter exceeded limits! Forcing fallback safety refusal.")
        return "handle_medical_refusal"
    
    # 2. Otherwise return standard arrays
    return state["required_agents"]

def evaluate_safety_gate(state: WellnessState) -> str:
    """Evaluates whether the safety audit passed or requires a medical disclaimer."""
    status = state.get("safety_status", "").upper()
    if "CRITICAL REJECTION" in status:
        print("❌ [Safety Gate] Plan REJECTED by auditor. Redirecting to medical disclaimer.")
        return "handle_medical_refusal"
    print("✅ [Safety Gate] Plan PASSED compliance audit. Finalizing layout output.")
    return "finalize_and_save"

# =========================================================
# 4. Compile the High-Level Orchestration Graph
# =========================================================
workflow = StateGraph(WellnessState)

# Add all runtime nodes
workflow.add_node("initialize", initialize_workflow_node)
workflow.add_node("intent_analyzer", intent_analyzer_node)
workflow.add_node("trainer", trainer_node)
workflow.add_node("yogi", yogi_node)
workflow.add_node("dietitian", dietitian_node)
workflow.add_node("safety_audit", safety_audit_node)
workflow.add_node("handle_medical_refusal", handle_medical_refusal_node)
workflow.add_node("finalize_and_save", finalize_and_save_node)

# Set the Entry Point pipeline
workflow.add_edge(START, "initialize")
workflow.add_edge("initialize", "intent_analyzer")

# Parallel routing conditional configuration
workflow.add_conditional_edges(
    "intent_analyzer",
    route_to_agents,
    {
        "trainer": "trainer",
        "yogi": "yogi",
        "dietitian": "dietitian",
        "handle_medical_refusal": "handle_medical_refusal" # Explicitly mapped to support circuit breaker exit path!
    }
)

# Join paths smoothly at the safety audit node
workflow.add_edge("trainer", "safety_audit")
workflow.add_edge("yogi", "safety_audit")
workflow.add_edge("dietitian", "safety_audit")

# Conditional safety audit routing layout gate
workflow.add_conditional_edges(
    "safety_audit",
    evaluate_safety_gate,
    {
        "handle_medical_refusal": "handle_medical_refusal",
        "finalize_and_save": "finalize_and_save"
    }
)

# Connect terminal paths cleanly to the end node
workflow.add_edge("handle_medical_refusal", END)
workflow.add_edge("finalize_and_save", END)

wellness_orchestrator = workflow.compile()

def execute_wellness_orchestration(user_id: str, user_message: str) -> str:
    initial_inputs = {
        "user_id": user_id, 
        "user_message": user_message, 
        "required_agents": [],
        "loop_counter": 0 # Explicit initialization
    }
    final_state = wellness_orchestrator.invoke(initial_inputs)
    return final_state["final_output"]