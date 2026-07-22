import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from tools import search_youtube_videos, search_and_scrape_recipe

# =========================================================
# 1. Initialize Core Models (No Anthropic / Pure Gemini + Groq)
# =========================================================
specialist_flash_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.4,
    max_retries=1,
    timeout=30
)

analytical_pro_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0,
    max_retries=1
)

# Use Gemini 2.5 Flash for general chat if GOOGLE_API_KEY exists, otherwise fall back to Groq
google_key = os.getenv("GOOGLE_API_KEY")
if google_key and not google_key.startswith("YOUR_"):
    conversational_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_key,
        temperature=0.7,
        max_output_tokens=1000
    )
else:
    conversational_model = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
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
# 3. System Prompts
# =========================================================
TRAINER_PROMPT = """You are an experienced strength and conditioning coach talking directly to a client — confident, direct, practical, and highly knowledgeable.

Provide:
1. A quick read on their situation based on their request and profile.
2. 1-2 concrete exercises with actionable coaching form cues.
3. Common pitfalls to avoid to prevent injury.

Never invent video links — only include real markdown links if returned directly from a tool call."""

YOGI_PROMPT = """You are an expert yoga and mobility instructor — calm, precise, and practical.

Provide:
1. A brief physical read on the targeted muscle group or joint mechanics.
2. 1-2 specific poses with clear alignment cues.
3. Precise contraindications to protect against strain.

Never invent video links — only include real markdown links if returned directly from a tool call."""

DIETITIAN_PROMPT = """You are a sports dietitian — practical, evidence-based, and clear.

Provide actionable guidance focusing on macro balance, calorie timing, or meal composition tailored to the user's needs.

Never invent recipe details — only cite specifics if returned directly from a tool call."""

SAFETY_PROMPT = """You are a Health Safety Auditor.
If the generated plan is safe and compliant, output: COMPLIANCE PASSED.
If there is a direct physical hazard or dangerous health instruction, output: CRITICAL REJECTION followed by a short explanation."""

GENERAL_CHAT_PROMPT = """You are an intelligent, articulate, and friendly AI assistant who specializes in health and wellness.

Guidelines:
1. OUT-OF-BOUNDS / GENERAL QUESTIONS: Answer general knowledge queries, science questions, coding, math, history, or casual small-talk thoroughly and accurately like a top-tier general LLM. Do NOT refuse general queries or add disclaimers about being strictly a fitness bot.
2. WELLNESS & GREETINGS: Be warm, engaging, and practical.
3. CONTEXT: Integrate user profile details or past message history naturally if applicable.

User Profile:
{profile}

Recent History:
{history}

User Message:
{message}
"""

# =========================================================
# 4. Bind Tools & Prompts
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
# 5. Agent Execution Interfaces
# =========================================================
def run_trainer_agent(user_profile: str, user_message: str) -> str:
    chain = trainer_prompt | trainer_engine
    response = chain.invoke({"profile": user_profile, "user_message": user_message})
    
    if hasattr(response, "tool_calls") and response.tool_calls:
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
        
    return parse_llm_output(response.content)

def run_yogi_agent(user_profile: str, user_message: str, workout_plan: str) -> str:
    chain = yogi_prompt | yogi_engine
    response = chain.invoke({"profile": user_profile, "user_message": user_message, "workout": workout_plan})
    
    if hasattr(response, "tool_calls") and response.tool_calls:
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
        
    return parse_llm_output(response.content)

def run_dietitian_agent(user_profile: str, user_message: str, workload: str) -> str:
    chain = dietitian_prompt | dietitian_engine
    response = chain.invoke({"profile": user_profile, "user_message": user_message, "workload": workload})
    
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_call = response.tool_calls[0]
        try:
            tool_output = search_and_scrape_recipe.invoke(tool_call["args"])
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

    return parse_llm_output(response.content)

def run_safety_agent(user_profile: str, complete_plan: str) -> str:
    if "?" in complete_plan and len(complete_plan) < 250:
        return "COMPLIANCE PASSED"

    chain = safety_prompt_template | analytical_pro_model
    response = chain.invoke({"profile": user_profile, "plan": complete_plan})
    return parse_llm_output(response.content)

def run_general_chat_agent(user_message: str, recent_history: str = "", user_profile: str = "") -> str:
    prompt = GENERAL_CHAT_PROMPT.format(
        profile=user_profile or "Not available",
        history=recent_history or "No prior history",
        message=user_message
    )
    response = conversational_model.invoke(prompt)
    return parse_llm_output(response.content)
