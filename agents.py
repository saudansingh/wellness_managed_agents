import os
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from tools import search_youtube_videos, search_and_scrape_recipe

# =========================================================
# 1. Initialize Core Models
# =========================================================
specialist_flash_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
    max_retries=1, # Bypasses systemic tenacity freezes
    timeout=30     # Cut down timeout window
)

# The Intellectual Engine: Shared utility instance
analytical_pro_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0,
    max_retries=1
)

# =========================================================
# 2. Output Cleaning Helper
# =========================================================
def parse_llm_output(raw_output) -> str:
    if not raw_output:
        return ""
    if isinstance(raw_output, str):
        return raw_output.strip()
    if isinstance(raw_output, dict) and "output" in raw_output:
        return str(raw_output["output"]).strip()
    return str(raw_output)

# =========================================================
# 3. System Prompts with Dynamic Clarification Gates
# =========================================================

TRAINER_PROMPT = """You are an elite, no-nonsense Personal Trainer and Strength Coach. 

CRITICAL INITIAL CHECK:
Review the user's profile and message history. If you do not know their specific training target (e.g., muscle gain, weight loss, conditioning) or equipment availability, do NOT output a plan. Instead, ask them 1-2 sharp, friendly clarifying questions to get those metrics.

IF YOU HAVE ALL THE DATA, STRUCTURE YOUR TOTAL RESPONSE INTO EXACTLY 4 PARTS:
1. DIAGNOSTIC: A 2-sentence direct assessment of their problem or goal.
2. WHAT TO DO & HOW TO DO IT: Provide 2 specific exercises. For each, give its name, its target focus, and sharp 1-step form instruction.
3. VIDEO COACHING ASSISTANTS: If video links are available from a tool call, present 2 or 3 exact markdown video links like this: * [Coach Instruction: Video Title](URL). Otherwise, give concise form layout advice.
4. THE ABSOLUTE DONT'S: Provide exactly 2 critical safety warnings stating what NOT to do.

Keep descriptions concise and highly tactical."""

YOGI_PROMPT = """You are a traditional Yoga Guru and Alignment Specialist. 

CRITICAL INITIAL CHECK:
Review the user's profile and message history. If you do not know what areas are tight, stiff, or their structural flexibility limits, do NOT output a plan. Instead, politely ask them to clarify what they want to address first.

IF YOU HAVE ALL THE DATA, STRUCTURE YOUR TOTAL RESPONSE INTO EXACTLY 4 PARTS:
1. GURU'S DISCERNMENT: A 2-sentence physical perspective on balancing the targeted area.
2. THE ASANA SEQUENCE: Provide 2 specific poses. State the pose name, what it stabilizes/stretches, and 1 precise alignment cue.
3. GUIDED PRACTICE VIDEOS: Present exact markdown video links if tools were run, else provide focused somatic adjustments.
4. CRITICAL RESTRICTIONS: Provide exactly 2 warnings on what NOT to do to protect the spine and joints."""

DIETITIAN_PROMPT = """You are a Clinical Sports Dietitian. 

CRITICAL INITIAL CHECK (CROSS-QUESTIONING GATE):
Review the user's profile and text inputs. If you do not know their specific dietary preferences (e.g., Vegetarian, Non-Vegetarian, Vegan, Eggetarian) or their primary objective (e.g., gaining weight, losing fat, maintaining), you MUST NOT build a meal structure yet.
Stop immediately and ask them nicely but directly to clarify (e.g., "Before I compile your macro breakdown, are you looking to gain weight or lose fat? Also, do you follow a Vegetarian or Non-Vegetarian diet?").

IF YOU HAVE ALL the metrics above, STRUCTURE YOUR TOTAL RESPONSE INTO EXACTLY 3 PARTS:
1. NUTRITIONAL ARCHITECTURE: A 2-sentence biochemical breakdown explaining how to fuel or recover for this exact goal.
2. WHAT TO EAT & PREPARATION: Provide 1 or 2 specific recipes or food options. State the exact macro benefit and a 1-sentence preparation tip.
3. METABOLIC DONT'S: Provide exactly 2 strict dietary rules outlining what foods, timings, or habits they must absolutely avoid."""

SAFETY_PROMPT = """You are a realistic Health Safety Auditor.
If the plan contains questions asking for more user information, or if it is compliant, output: COMPLIANCE PASSED.
If there is a direct physical hazard, output: CRITICAL REJECTION followed by a short description."""

# =========================================================
# 4. Bind Tools to Models
# =========================================================
trainer_engine = specialist_flash_model.bind_tools([search_youtube_videos])
yogi_engine = specialist_flash_model.bind_tools([search_youtube_videos])
dietitian_engine = specialist_flash_model.bind_tools([search_and_scrape_recipe])

trainer_prompt = ChatPromptTemplate.from_messages([
    ("system", TRAINER_PROMPT),
    ("human", "User Profile: {profile}\nUser Request: {user_message}")
])

yogi_prompt = ChatPromptTemplate.from_messages([
    ("system", YOGI_PROMPT),
    ("human", "User Profile: {profile}\nTrainer Context: {workout}\nUser Request: {user_message}")
])

dietitian_prompt = ChatPromptTemplate.from_messages([
    ("system", DIETITIAN_PROMPT),
    ("human", "User Profile: {profile}\nActivity Context: {workload}\nUser Request: {user_message}")
])

safety_prompt_template = ChatPromptTemplate.from_messages([
    ("system", SAFETY_PROMPT),
    ("human", "User Profile: {profile}\n\nGenerated Response:\n{plan}")
])

# =========================================================
# 5. Agent Execution Interfaces (With Performance Tuning)
# =========================================================

def run_trainer_agent(user_profile: str, user_message: str) -> str:
    # High-speed condition: Only trigger slow tools if they explicitly ask for videos/links
    msg_lower = user_message.lower()
    if "video" in msg_lower or "watch" in msg_lower or "link" in msg_lower:
        chain = trainer_prompt | trainer_engine
        response = chain.invoke({"profile": user_profile, "user_message": user_message})
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_output = search_youtube_videos.invoke(tool_call["args"])
            final_prompt = ChatPromptTemplate.from_messages([
                ("system", TRAINER_PROMPT),
                ("human", "User Profile: {profile}\nUser Request: {user_message}"),
                response,
                ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
            ])
            final_response = (final_prompt | specialist_flash_model).invoke({"profile": user_profile, "user_message": user_message})
            return parse_llm_output(final_response.content)
    else:
        # Instant completion bypasses tool calls entirely
        fast_chain = trainer_prompt | specialist_flash_model
        response = fast_chain.invoke({"profile": user_profile, "user_message": user_message})
    
    return parse_llm_output(response.content)

def run_yogi_agent(user_profile: str, user_message: str, workout_plan: str) -> str:
    msg_lower = user_message.lower()
    if "video" in msg_lower or "routine" in msg_lower:
        chain = yogi_prompt | yogi_engine
        response = chain.invoke({"profile": user_profile, "user_message": user_message, "workout": workout_plan})
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_output = search_youtube_videos.invoke(tool_call["args"])
            final_prompt = ChatPromptTemplate.from_messages([
                ("system", YOGI_PROMPT),
                ("human", "User Profile: {profile}\nTrainer Context: {workout}\nUser Request: {user_message}"),
                response,
                ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
            ])
            final_response = (final_prompt | specialist_flash_model).invoke({"profile": user_profile, "workout": workout_plan, "user_message": user_message})
            return parse_llm_output(final_response.content)
    else:
        fast_chain = yogi_prompt | specialist_flash_model
        response = fast_chain.invoke({"profile": user_profile, "workout": workout_plan, "user_message": user_message})
        
    return parse_llm_output(response.content)

def run_dietitian_agent(user_profile: str, user_message: str, workload: str) -> str:
    msg_lower = user_message.lower()
    
    # Check if the user is missing crucial info first. If they are, bypass recipe search tool to instantly respond with questions.
    is_missing_info = "veg" not in msg_lower and "meat" not in msg_lower and "gain" not in msg_lower and "lose" not in msg_lower
    
    if ("recipe" in msg_lower or "cook" in msg_lower or "eat" in msg_lower) and not is_missing_info:
        chain = dietitian_prompt | dietitian_engine
        response = chain.invoke({"profile": user_profile, "user_message": user_message, "workload": workload})
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            try:
                tool_output = search_and_scrape_recipe.invoke(tool_call["args"])
                if not tool_output or "Error" in str(tool_output):
                    tool_output = "Provide a high-calorie nutrition structure focusing on complex carbs and lean proteins."
                final_prompt = ChatPromptTemplate.from_messages([
                    ("system", DIETITIAN_PROMPT),
                    ("human", "User Profile: {profile}\nActivity Context: {workload}\nUser Request: {user_message}"),
                    response,
                    ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                ])
                final_response = (final_prompt | specialist_flash_model).invoke({"profile": user_profile, "workload": workload, "user_message": user_message})
                return parse_llm_output(final_response.content)
            except Exception:
                pass

    # Instant execution fallback when tools aren't explicitly needed or if info needs cross-questioning
    fallback_chain = dietitian_prompt | specialist_flash_model
    final_response = fallback_chain.invoke({"profile": user_profile, "workload": workload, "user_message": user_message})
    return parse_llm_output(final_response.content)

def run_safety_agent(user_profile: str, complete_plan: str) -> str:
    # Performance Optimization: If response is a plain conversational question, skip heavy evaluation entirely
    if "?" in complete_plan and len(complete_plan) < 250:
        return "COMPLIANCE PASSED"
        
    chain = safety_prompt_template | analytical_pro_model
    response = chain.invoke({"profile": user_profile, "plan": complete_plan})
    return parse_llm_output(response.content)
