import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from tools import search_youtube_videos, search_and_scrape_recipe

# =========================================================
# 1. Initialize Core Models
# =========================================================
specialist_flash_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.4,
    max_retries=1,
    timeout=20
)

analytical_pro_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0,
    max_retries=1
)

google_key = os.getenv("GOOGLE_API_KEY")
if google_key and not google_key.startswith("YOUR_"):
    conversational_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_key,
        temperature=0.6,
        max_output_tokens=350  # Capped for snappy, brief responses
    )
else:
    conversational_model = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.6,
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
# 3. Streamlined System Prompts (Strict Brevity & Flow)
# =========================================================
TRAINER_PROMPT = """You are an expert strength coach. Be direct, punchy, and concise (under 150 words).

Provide:
1. Quick assessment.
2. 1-2 exercises with form cues.
3. 1 pitfall to avoid.

Never invent video links — only include markdown links if returned by a tool call."""

YOGI_PROMPT = """You are a mobility & yoga instructor. Be calm, practical, and concise (under 150 words).

Provide:
1. Brief physical focus.
2. 1-2 poses with alignment cues.
3. Contraindications.

Never invent video links — only include markdown links if returned by a tool call."""

DIETITIAN_PROMPT = """You are a sports dietitian. Be practical and concise (under 150 words).

Provide key macro/dietary guidance tailored to the user.
Never invent recipe details — only cite specifics if returned by a tool call."""

SAFETY_PROMPT = """You are a Health Safety Auditor.
If the generated plan is safe, output: COMPLIANCE PASSED.
If dangerous, output: CRITICAL REJECTION followed by a short explanation."""

GENERAL_CHAT_PROMPT = """You are a friendly, intelligent AI health & wellness assistant.

CRITICAL CHAT RULES:
1. NO REPETITIVE GREETINGS: Do NOT say "Welcome again", "Welcome back", "Hello!", or re-introduce yourself if there is existing dialogue history. Treat this as an ongoing, continuous chat.
2. BREVITY: Keep answers concise, direct, and under 100 words for simple chats unless the user explicitly requests an in-depth explanation.
3. GENERAL QUESTIONS: Answer non-wellness queries (math, coding, general science) directly without disclaimers.

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
    if len(complete_plan) < 200:
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
