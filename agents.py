import os
import ast
import re
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List, Optional

# Import the direct executable tools safely
from tools import search_youtube_videos, search_and_scrape_recipe

# =========================================================
# 1. Initialize Core Models (Optimized for Speed and Logic)
# =========================================================

# Look for standard environment keys mapped out via .env load cycles
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_SEARCH_API_KEY")

if not API_KEY:
    raise ValueError("❌ Error: Missing API key. Ensure GOOGLE_SEARCH_API_KEY is defined inside your .env configuration.")

# The Speed Engine: Lowered retries to fail fast and avoid the 4-minute hang loop
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
# 2. Output Cleaning Helper Function
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
# 3. Systematic Agent Architecture System Prompts
# =========================================================

TRAINER_PROMPT = """You are an elite, no-nonsense Personal Trainer and Strength Coach. 
Address the client directly with authority, clarity, and an encouraging yet strict professional tone.

STRUCTURE YOUR TOTAL RESPONSE INTO EXACTLY 4 PARTS:
1. DIAGNOSTIC: A 2-sentence direct assessment of their problem or goal.
2. WHAT TO DO & HOW TO DO IT: Provide 2 specific exercises. For each, give its name, its target focus, and sharp 1-step form instruction.
3. VIDEO COACHING ASSISTANTS: Present 2 or 3 exact markdown video links retrieved from your tool like this:
   * [Coach Instruction: Video Title](URL)
4. THE ABSOLUTE DONT'S: Provide exactly 2 critical safety warnings stating what NOT to do.

Keep descriptions concise and highly tactical—like a coach talking to an athlete on the gym floor."""

YOGI_PROMPT = """You are a traditional Yoga Guru and Alignment Specialist. 
Speak to the practitioner with mindful clarity, deep anatomical wisdom, and absolute precision.

STRUCTURE YOUR TOTAL RESPONSE INTO EXACTLY 4 PARTS:
1. GURU'S DISCERNMENT: A 2-sentence spiritual and physical perspective on balancing the targeted area.
2. THE ASANA SEQUENCE: Provide 2 specific poses. State the pose name, what it stabilizes/stretches, and 1 precise alignment cue.
3. GUIDED PRACTICE VIDEOS: Present 2 or 3 exact markdown video links retrieved from your tool like this:
   * [Guru Practice: Video Title](URL)
4. CRITICAL RESTRICTIONS: Provide exactly 2 warnings on what NOT to do to protect the spine and joints.

Avoid fluff; deliver raw therapeutic yoga protocol."""

DIETITIAN_PROMPT = """You are a Clinical Sports Dietitian. 
Your tone is scientific, exact, practical, and highly prescriptive. 

STRUCTURE YOUR TOTAL RESPONSE INTO EXACTLY 3 PARTS:
1. NUTRITIONAL ARCHITECTURE: A 2-sentence biochemical breakdown explaining how to fuel or recover for this exact goal.
2. WHAT TO EAT & PREPARATION: Provide 1 or 2 specific recipes or ingredient structures retrieved from your tool. State the exact macro benefit and a 1-sentence cooking tip.
3. METABOLIC DONT'S: Provide exactly 2 strict dietary rules outlining what foods, timings, or habits they must absolutely avoid.

Deliver an elite nutrition strategy without filler sentences."""

SAFETY_PROMPT = """You are a realistic Health Safety Auditor for an adaptive fitness application. 
Your task is to cross-examine the generated plan against the user's physical profile parameters.

CRITICAL INSTRUCTION:
Do NOT issue a rejection for standard lifestyle modifications, general caloric advice, healthy meal options, or gentle therapeutic stretching routines unless they directly and explicitly endanger the user.

OUTPUT FORMAT SPECIFICATION:
- If the plan is safe, helpful, and reasonable, your output MUST be exactly the string: COMPLIANCE PASSED
- If there is a clear, dangerous physical hazard, your output MUST begin with: CRITICAL REJECTION followed by a clear, short description of the direct hazard.
"""

# =========================================================
# 4. Bind Tools Directly to the Google GenAI Engine
# =========================================================
# We bind the tools directly to the model run configurations, bypassing messy legacy executors.
trainer_engine = specialist_flash_model.bind_tools([search_youtube_videos])
yogi_engine = specialist_flash_model.bind_tools([search_youtube_videos])
dietitian_engine = specialist_flash_model.bind_tools([search_and_scrape_recipe])

# Build Clean Declarative Prompts
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
    # 1. Ask the model for intent and tool calls
    chain = trainer_prompt | trainer_engine
    response = chain.invoke({"profile": user_profile, "user_message": user_message})
    
    # 2. If the model wants to search YouTube, execute it immediately right here!
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        # Execute the Python function directly
        tool_output = search_youtube_videos.invoke(tool_call["args"])
        
        # Feed the tool results back to the model so it reads the links and formats your short answer
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", TRAINER_PROMPT),
            ("human", "User Profile: {profile}\nUser Request: {user_message}"),
            response, # original model message with tool requests
            ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
        ])
        
        final_response = (final_prompt | specialist_flash_model).invoke({
            "profile": user_profile, 
            "user_message": user_message
        })
        return parse_llm_output(final_response.content)
        
    return parse_llm_output(response.content)

def run_yogi_agent(user_profile: str, user_message: str, workout_plan: str) -> str:
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
        final_response = (final_prompt | specialist_flash_model).invoke({
            "profile": user_profile, 
            "workout": workout_plan, 
            "user_message": user_message
        })
        return parse_llm_output(final_response.content)
        
    return parse_llm_output(response.content)

def run_dietitian_agent(user_profile: str, user_message: str, workload: str) -> str:
    # 1. Ask the dietitian model for intent and tool execution
    chain = dietitian_prompt | dietitian_engine
    response = chain.invoke({"profile": user_profile, "user_message": user_message, "workload": workload})
    
    # 2. Check if the model wants to call its scraping/recipe tool
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        try:
            # Execute the scraping python function safely
            tool_output = search_and_scrape_recipe.invoke(tool_call["args"])
            
            # If the tool fails or comes back empty, provide clean fallback context so it doesn't crash
            if not tool_output or "Error" in str(tool_output):
                tool_output = "Provide a high-calorie, nutrient-dense muscle building structure focusing on complex carbs and lean proteins."
            
            # Feed data back to model for the final structured professional response
            final_prompt = ChatPromptTemplate.from_messages([
                ("system", DIETITIAN_PROMPT),
                ("human", "User Profile: {profile}\nActivity Context: {workload}\nUser Request: {user_message}"),
                response,
                ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
            ])
            
            final_response = (final_prompt | specialist_flash_model).invoke({
                "profile": user_profile,
                "workload": workload,
                "user_message": user_message
            })
            return parse_llm_output(final_response.content)
            
        except Exception:
            # Safe absolute fallback if the tool step completely unbraids
            fallback_chain = dietitian_prompt | specialist_flash_model
            final_response = fallback_chain.invoke({"profile": user_profile, "workload": workload, "user_message": user_message})
            return parse_llm_output(final_response.content)
            
    # If no tool call was initialized, return regular output
    return parse_llm_output(response.content)

def run_safety_agent(user_profile: str, complete_plan: str) -> str:
    chain = safety_prompt_template | analytical_pro_model
    response = chain.invoke({"profile": user_profile, "plan": complete_plan})
    return parse_llm_output(response.content)