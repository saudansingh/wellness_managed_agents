import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from tools import search_youtube_videos, search_and_scrape_recipe

# =========================================================
# 1. Initialize Models
# =========================================================
specialist_flash_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
    max_retries=1,
    timeout=15
)

google_key = os.getenv("GOOGLE_API_KEY")
if google_key and not google_key.startswith("YOUR_"):
    conversational_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_key,
        temperature=0.5,
        max_output_tokens=250  # Hard token limit for concise answers
    )
else:
    conversational_model = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.5,
        max_retries=1
    )

def parse_llm_output(raw_output) -> str:
    if not raw_output:
        return ""
    if isinstance(raw_output, str):
        return raw_output.strip()
    if isinstance(raw_output, dict) and "output" in raw_output:
        return str(raw_output["output"]).strip()
    return str(raw_output)

# =========================================================
# 2. Strict & Ultra-Lean System Prompts
# =========================================================
TRAINER_PROMPT = """You are a top strength coach. Keep responses under 80 words.
Provide 1 key exercise, 1 form cue, and 1 safety tip.
Never invent video links — only include markdown links if returned directly by a tool call."""

YOGI_PROMPT = """You are a mobility instructor. Keep responses under 80 words.
Provide 1-2 stretches/poses with alignment cues and contraindications.
Never invent video links — only include markdown links if returned directly by a tool call."""

DIETITIAN_PROMPT = """You are a sports dietitian. Keep responses under 80 words.
Provide practical, direct nutrition advice or macro guidance tailored to the request.
Never invent recipe details — only cite specifics if returned directly by a tool call."""

GENERAL_CHAT_PROMPT = """You are an intelligent, versatile AI assistant (like ChatGPT or Gemini).

STRICT RESPONSE RULES:
1. ANY TOPIC ALLOWED: Answer general knowledge, coding, math, science, history, casual chit-chat, or wellness questions thoroughly and accurately.
2. NO BOT DISCLAIMERS: Never say "As a fitness bot...", "I am a health assistant...", or add unnecessary disclaimers. Answer directly.
3. NO FILLER GREETINGS: NEVER say "Welcome back", "Hello again", "Welcome!", or re-introduce yourself. Jump straight into answering the user's question.
4. MAXIMUM BREVITY: Keep answers punchy and short (2 to 4 sentences or concise bullet points) unless the user explicitly asks for an essay, guide, or code code snippet.

User Profile: {profile}
Recent Dialogue History: {history}
User Query: {message}"""

# =========================================================
# 3. Agent Execution Interfaces
# =========================================================
trainer_engine = specialist_flash_model.bind_tools([search_youtube_videos])
yogi_engine = specialist_flash_model.bind_tools([search_youtube_videos])
dietitian_engine = specialist_flash_model.bind_tools([search_and_scrape_recipe])

def run_trainer_agent(user_profile: str, user_message: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", TRAINER_PROMPT),
        ("human", "Profile: {profile}\nQuery: {user_message}")
    ])
    response = (prompt | trainer_engine).invoke({"profile": user_profile, "user_message": user_message})
    
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_output = search_youtube_videos.invoke(tool_call["args"])
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", TRAINER_PROMPT),
            ("human", "Profile: {profile}\nQuery: {user_message}"),
            response,
            ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
        ])
        final_response = (final_prompt | specialist_flash_model).invoke({"profile": user_profile, "user_message": user_message})
        return parse_llm_output(final_response.content)
        
    return parse_llm_output(response.content)

def run_yogi_agent(user_profile: str, user_message: str, workout_plan: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", YOGI_PROMPT),
        ("human", "Profile: {profile}\nWorkout Context: {workout}\nQuery: {user_message}")
    ])
    response = (prompt | yogi_engine).invoke({"profile": user_profile, "user_message": user_message, "workout": workout_plan})
    return parse_llm_output(response.content)

def run_dietitian_agent(user_profile: str, user_message: str, workload: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", DIETITIAN_PROMPT),
        ("human", "Profile: {profile}\nWorkload Context: {workload}\nQuery: {user_message}")
    ])
    response = (prompt | dietitian_engine).invoke({"profile": user_profile, "user_message": user_message, "workload": workload})
    return parse_llm_output(response.content)

def run_general_chat_agent(user_message: str, recent_history: str = "", user_profile: str = "") -> str:
    prompt = GENERAL_CHAT_PROMPT.format(
        profile=user_profile or "None",
        history=recent_history or "None",
        message=user_message
    )
    response = conversational_model.invoke(prompt)
    return parse_llm_output(response.content)
