import re
import random
from typing import TypedDict, Union, Optional, List, Any
from langgraph.graph import StateGraph, START, END

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
    get_recent_history,
    save_chat_message
)

# Canonical execution order for when more than one specialist is needed, so
# later agents actually see earlier agents' real output instead of racing
# them in parallel.
AGENT_ORDER = ["trainer", "yogi", "dietitian"]

# =========================================================
# 1. Shared Graph State
# =========================================================
class WellnessState(TypedDict):
    user_id: str
    user_message: str
    user_profile: Optional[str]
    recent_history: Optional[str]

    required_agents: List[str]
    agent_queue: List[str]

    workout_plan: Optional[str]
    yoga_plan: Optional[str]
    diet_plan: Optional[str]

    safety_status: Optional[str]
    final_output: Optional[str]

    week_number: Optional[int]
    _fast_route: Optional[str]

# =========================================================
# 2. Fast Router — pure Python, zero I/O
# =========================================================
# This is the ONLY thing that runs for every single message. It is a plain
# regex/keyword classifier with no network calls (no LLM, no DB), so it
# resolves in microseconds. It decides one of three paths:
#   - "unsafe"       -> jailbreak/prompt-extraction attempt, canned refusal
#   - "specialists"  -> clear fitness/yoga/nutrition request, keyword-matched
#   - "chat"         -> everything else (greetings, small talk, off-topic,
#                        vague messages, ambiguous wellness phrasing) -> one
#                        adaptive LLM call in general_chat_node, no DB hit.

GREETING_RE = re.compile(
    r"^\s*(hi+|hello+|hey+|yo|sup|good\s?(morning|afternoon|evening)|"
    r"what'?s up|howdy|namaste)\s*[!.,?]*\s*$",
    re.IGNORECASE
)

UNSAFE_RE = re.compile(
    r"(ignore (all|previous|above) instructions|disregard (all|previous|above)|"
    r"system prompt|reveal.*(prompt|instructions)|jailbreak|api[\s_-]?key|"
    r"you are now|pretend you are|dan mode)",
    re.IGNORECASE
)

TRAINER_KW = {
    "workout", "workouts", "exercise", "exercises", "gym", "muscle", "muscles",
    "strength", "reps", "sets", "lift", "lifting", "cardio", "training",
    "pushup", "push-up", "pushups", "squat", "squats", "deadlift", "fitness",
    "bicep", "biceps", "abs", "core", "hiit"
}
YOGI_KW = {
    "yoga", "stretch", "stretching", "flexibility", "asana", "pose", "poses",
    "mobility", "meditation", "breathing", "pranayama", "spine", "posture"
}
DIET_KW = {
    "diet", "food", "recipe", "recipes", "meal", "meals", "calorie", "calories",
    "macro", "macros", "nutrition", "protein", "carb", "carbs", "eat", "eating",
    "vegan", "vegetarian", "keto", "supplement", "supplements"
}

GREETING_REPLIES = [
    "Hey there! 👋 Want help with a workout, yoga session, or your nutrition plan today?",
    "Hi! What can I help you with — training, yoga, or diet?",
    "Hello! I'm your wellness assistant. Fitness, yoga, or nutrition — where do we start?",
]

def fast_router_node(state: WellnessState) -> dict:
    message = state["user_message"].strip()
    lower = message.lower()

    if UNSAFE_RE.search(lower):
        return {"_fast_route": "unsafe"}

    if GREETING_RE.match(lower):
        return {"_fast_route": "chat_instant", "final_output": random.choice(GREETING_REPLIES)}

    words = set(re.findall(r"[a-z]+", lower))
    found = set()
    if words & TRAINER_KW:
        found.add("trainer")
    if words & YOGI_KW:
        found.add("yogi")
    if words & DIET_KW:
        found.add("dietitian")

    if found:
        required = [a for a in AGENT_ORDER if a in found]
        return {"_fast_route": "specialists", "required_agents": required, "agent_queue": required.copy()}

    # No keyword match, not a greeting, not unsafe -> anything else at all:
    # small talk, vague replies, genuinely off-topic questions, or wellness
    # phrasing that didn't hit a keyword. One adaptive LLM call handles all
    # of it in general_chat_node, no DB round trip needed.
    return {"_fast_route": "chat"}

# =========================================================
# 3. Graph Nodes
# =========================================================

def initialize_workflow_node(state: WellnessState) -> dict:
    """Only reached by the 'specialists' path — general chat / greetings /
    unsafe never touch the database at all."""
    user_id = state["user_id"]
    profile_str = get_user_profile_string(user_id)
    current_max_week = get_last_week_number(user_id)
    history_context = get_recent_history(user_id, turns=6) or ""

    full_profile_context = profile_str
    if history_context:
        full_profile_context = f"{profile_str}\n\nRecent conversation:\n{history_context}"

    return {
        "user_profile": full_profile_context,
        "recent_history": history_context,
        "week_number": current_max_week + 1,
        "workout_plan": "",
        "yoga_plan": "",
        "diet_plan": "",
        "safety_status": "",
        "final_output": ""
    }



def general_chat_node(state: WellnessState) -> dict:
    """
    Single adaptive handler for anything that isn't a clear specialist
    request: greetings that slipped past the instant regex, small talk,
    vague replies, and — per your requirement — genuinely out-of-bound
    questions. For out-of-bound questions this must politely explain the
    bot's scope and point the user elsewhere, rather than attempting an
    answer it has no data for.
    """
    print("💬 [Agent] General Chat activated.")
    chat_prompt = f"""You are a warm, natural AI Wellness Assistant. You ONLY have real expertise in
fitness training, yoga, and nutrition — you do not have access to any other live data
(no news, weather, general trivia lookups, coding help, etc).

Respond directly to what the user just said:
- If it's a greeting or casual chat, reply naturally and briefly (1-2 sentences), and you may
  mention you can help with workouts, yoga, or nutrition.
- If it's a genuine question outside your wellness scope (general knowledge, unrelated topics,
  anything you cannot reliably answer), politely say you're a wellness bot and don't have access
  to that kind of information, and suggest they check a relevant site/search engine for it.
  Keep it short and friendly, not robotic.
- If it's vague or unclear, ask a natural one-line follow-up instead of a fixed error message.
- Never fabricate factual information outside fitness/yoga/nutrition.

Keep the whole response to 1-3 sentences.

User's message: "{state['user_message']}"
Response:"""
    response = analytical_pro_model.invoke(chat_prompt).content.strip()
    return {"final_output": response}

def safe_redirect_node(state: WellnessState) -> dict:
    return {"final_output": "I can't override my safety instructions or share internal details, but I'm glad to help with your wellness goals."}

def handle_no_profile_node(state: WellnessState) -> dict:
    return {"final_output": (
        "It looks like you haven't completed your profile setup yet — I need your age, weight, "
        "and any injuries or goals to give you a safe, personalized plan. Please finish onboarding first!"
    )}

def trainer_node(state: WellnessState) -> dict:
    print("🏋️ [Agent] Trainer activated.")
    workout = run_trainer_agent(state["user_profile"], state["user_message"])
    return {"workout_plan": workout, "agent_queue": state.get("agent_queue", [])[1:]}

def yogi_node(state: WellnessState) -> dict:
    print("🧘 [Agent] Yogi activated.")
    workout_context = state.get("workout_plan") or "No workout context provided."
    yoga = run_yogi_agent(state["user_profile"], state["user_message"], workout_context)
    return {"yoga_plan": yoga, "agent_queue": state.get("agent_queue", [])[1:]}

def dietitian_node(state: WellnessState) -> dict:
    print("🥗 [Agent] Dietitian activated.")
    workout_context = state.get("workout_plan") or "No workout planned."
    yoga_context = state.get("yoga_plan") or "No yoga planned."
    workload = f"Workout splits:\n{workout_context}\n\nYoga recovery:\n{yoga_context}"
    diet = run_dietitian_agent(state["user_profile"], state["user_message"], workload)
    return {"diet_plan": diet, "agent_queue": state.get("agent_queue", [])[1:]}

def safety_audit_node(state: WellnessState) -> dict:
    print("🛡️ [Agent] Safety Auditor evaluating plan safety parameters...")
    combined_plan = "--- USER REQUEST RESPONSE ---\n\n"
    if state.get("workout_plan"):
        combined_plan += f"{state['workout_plan']}\n\n"
    if state.get("yoga_plan"):
        combined_plan += f"{state['yoga_plan']}\n\n"
    if state.get("diet_plan"):
        combined_plan += f"{state['diet_plan']}\n\n"

    audit_result = run_safety_agent(state["user_profile"], combined_plan)
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
# 4. Conditional Routing
# =========================================================

def route_after_fast_router(state: WellnessState) -> str:
    return state["_fast_route"]

def route_after_profile_gate(state: WellnessState) -> str:
    """Only specialist requests actually need a completed profile — chat/
    greeting/unsafe never reach this gate at all (see the edge map below)."""
    if "No profile found" in state.get("user_profile", ""):
        return "handle_no_profile"
    return state["agent_queue"][0]

def route_next_agent(state: WellnessState) -> str:
    queue = state.get("agent_queue", [])
    if queue:
        return queue[0]
    return "safety_audit"

def evaluate_safety_gate(state: WellnessState) -> str:
    status = state.get("safety_status", "").upper()
    if "CRITICAL REJECTION" in status:
        return "handle_medical_refusal"
    return "finalize_and_save"

# =========================================================
# 5. Compile the Graph
# =========================================================
workflow = StateGraph(WellnessState)

workflow.add_node("fast_router", fast_router_node)
workflow.add_node("initialize", initialize_workflow_node)
workflow.add_node("profile_gate", profile_gate_node)
workflow.add_node("handle_no_profile", handle_no_profile_node)
workflow.add_node("trainer", trainer_node)
workflow.add_node("yogi", yogi_node)
workflow.add_node("dietitian", dietitian_node)
workflow.add_node("safety_audit", safety_audit_node)
workflow.add_node("handle_medical_refusal", handle_medical_refusal_node)
workflow.add_node("general_chat", general_chat_node)
workflow.add_node("safe_redirect", safe_redirect_node)
workflow.add_node("finalize_and_save", finalize_and_save_node)

workflow.add_edge(START, "fast_router")

# The ONE routing decision, made with zero I/O. "specialists" is the only
# branch that goes on to touch the database. "chat_instant" (pure greeting)
# is already resolved with final_output set, so it goes straight to END.
workflow.add_conditional_edges(
    "fast_router",
    route_after_fast_router,
    {
        "specialists": "initialize",
        "chat": "general_chat",
        "chat_instant": END,
        "unsafe": "safe_redirect",
    }
)



workflow.add_conditional_edges(
    "initialize",
    route_after_profile_gate,
    {
        "handle_no_profile": "handle_no_profile",
        "trainer": "trainer",
        "yogi": "yogi",
        "dietitian": "dietitian",
    }
)

# Sequential specialist chain: each agent hands off to the next required one
# (or the safety audit), so later agents see real earlier output instead of
# a placeholder string.
agent_next_map = {"trainer": "trainer", "yogi": "yogi", "dietitian": "dietitian", "safety_audit": "safety_audit"}
workflow.add_conditional_edges("trainer", route_next_agent, agent_next_map)
workflow.add_conditional_edges("yogi", route_next_agent, agent_next_map)
workflow.add_conditional_edges("dietitian", route_next_agent, agent_next_map)

workflow.add_conditional_edges(
    "safety_audit",
    evaluate_safety_gate,
    {
        "handle_medical_refusal": "handle_medical_refusal",
        "finalize_and_save": "finalize_and_save"
    }
)

workflow.add_edge("handle_no_profile", END)
workflow.add_edge("handle_medical_refusal", END)
workflow.add_edge("finalize_and_save", END)
workflow.add_edge("general_chat", END)
workflow.add_edge("safe_redirect", END)

wellness_orchestrator = workflow.compile()

def execute_wellness_orchestration(user_id: str, user_message: str) -> str:
    cleaned_message = user_message.strip()
    if not cleaned_message:
        return "I didn't receive any message — could you type your question?"

    initial_inputs = {
        "user_id": user_id,
        "user_message": cleaned_message,
        "required_agents": [],
        "agent_queue": [],
    }
    final_state = wellness_orchestrator.invoke(initial_inputs)
    final_output = final_state["final_output"]

    # Chat history is only meaningful for logged, persistent conversations.
    # Still logging every turn (including chat/greeting) keeps get_recent_history
    # useful for future specialist calls in the same session.
    save_chat_message(user_id, "user", cleaned_message)
    save_chat_message(user_id, "assistant", final_output)

    return final_output
