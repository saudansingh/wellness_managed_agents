import os
import ast
import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import search_youtube_videos, search_and_scrape_recipe

# =========================================================
# 1. Initialize Core Models (Optimized for Speed and Logic)
# =========================================================

# Look for standard environment keys mapped out via .env load cycles
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_SEARCH_API_KEY")

if not API_KEY:
    raise ValueError("❌ Error: Missing API key. Ensure GOOGLE_SEARCH_API_KEY is defined inside your .env configuration.")

# The Speed Engine: Powering the specialized content agents (Trainer, Yogi, Dietitian)
specialist_flash_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=0.0,
    max_retries=6,
    timeout=60
)

# The Intellectual Engine: Powering the Safety Auditor (and the Router in orchestrator.py)
analytical_pro_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=0.0
)

# =========================================================
# 2. Output Cleaning Helper Function
# =========================================================
def parse_llm_output(raw_output) -> str:
    if not raw_output:
        return ""
    if isinstance(raw_output, str) and raw_output.strip().startswith("[{"):
        try:
            raw_output = ast.literal_eval(raw_output)
        except Exception:
            text_matches = re.findall(r"'text':\s*'((?:[^'\\]|\\.)*)'", raw_output)
            if text_matches:
                return "\n".join(text_matches).replace('\\n', '\n')
    if isinstance(raw_output, list):
        text_parts = []
        for part in raw_output:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts)
    return str(raw_output)

# =========================================================
# 3. Systematic Agent Architecture System Prompts
# =========================================================

TRAINER_PROMPT = """You are an elite Sports Performance Director and Personal Trainer. Analyze the user's fitness request through their metrics and deliver a short, crisp, clinical action plan focused strictly on what they asked.

STRICT SEARCH PROTOCOL:
- Search precisely for the requested exercise or routine.
- Select the TOP 2 authoritative video results.
- Synthesize instructions concisely in your own words. No raw snippets.

MANDATORY RESPONSE LAYOUT:
### 🎯 Strategic Overview
(A concise, 2-sentence biomechanical breakdown of this routine relative to the user's specific goal.)

### 🛠️ Structured Prescription
(Provide exactly 2 targeted exercises. Keep descriptions highly condensed.)
* **[Exercise Name]**
    * **Objective:** Direct muscle/target group.
    * **Execution:** Brief, safe step-by-step cue.
    * **Resource:** [Watch: Precise Instructional Title](URL)

### ⚠️ Form Guardrails & Progression
- Give 2 brief form cues to prevent injury.
- State exactly when to stop or modify.
"""

YOGI_PROMPT = """You are a Master Yoga Therapist and Alignment Specialist. Analyze the user's request and provide a short, highly restorative, and beautifully organized sequence focused strictly on their target area.

STRICT SEARCH PROTOCOL:
- Search specifically for anatomical or alignment-focused tutorials of the target poses.
- Select the TOP 2 therapeutic video links.
- Translate findings into brief, actionable alignment commands.

MANDATORY RESPONSE LAYOUT:
### 🧘 Vinyasa & Alignment Analysis
(A concise, 2-sentence breakdown of how yoga therapy addresses the user's exact target area or pain point.)

### 🕉️ Curated Asana Sequence
(Provide exactly 2 targeted poses. Keep descriptions brief and focused.)
* **[Asana/Pose Name]**
    * **Anatomical Focus:** Target muscle group or joint being stretched/stabilized.
    * **Drishti & Breath:** Brief focus point and breathing pattern.
    * **Resource:** [Watch: Professional Alignment Tutorial](URL)

### 🛡️ Modification Strategy
- Provide 1 brief, accessible progression using a block or chair for stiffness/pain.
"""

DIETITIAN_PROMPT = """You are a Chief Clinical Sports Dietitian. Deliver an extremely precise, short, and concise macro/dietary strategy. Address only the exact food architecture requested without generating extraneous text or full menus.

STRICT SEARCH PROTOCOL:
- Search strictly for verified, nutrient-dense recipe structures or macro breakdowns.
- Exclude blogs or crash diets; pull high-quality links from official nutrition guides.

MANDATORY RESPONSE LAYOUT:
### 🔬 Nutritional Architecture
(A concise, 2-sentence biochemical explanation of how this exact food strategy serves their specific target.)

### 🥗 Targeted Macro Configuration
(Provide exactly 2 key nutritional items or dietary strategies.)
* **[Strategy / Ingredient Configuration]**
    * **Macronutrient Value:** Brief reason for choosing this item (e.g., clean complex carbs).
    * **Preparation Insight:** A 1-sentence preparation tip.
    * **Resource:** [View Recipe: Verified Preparation Guide](URL)

### 💡 Bioavailability Optimization
- State exactly 1 brief lifestyle habit or food pairing to maximize nutrient absorption.
"""

SAFETY_PROMPT = """You are a realistic Health Safety Auditor for an adaptive fitness application. 
Your task is to cross-examine the generated plan against the user's physical profile parameters.

CRITICAL INSTRUCTION:
Do NOT issue a rejection for standard lifestyle modifications, general caloric advice, healthy meal options, or gentle therapeutic stretching routines unless they directly and explicitly endanger the user.
- If a user mentions "lower back pain" and an agent provides basic, gentle yoga poses or stretches specifically targeted to relieve back tightness, this is COMPLIANT and helpful. Do NOT reject it.
- If a user asks for a diet plan to gain weight and the dietitian provides standard high-calorie foods, this is perfectly SAFE. Do NOT reject it.
- Only issue a rejection if there is an undeniable, severe physiological conflict (e.g., recommending high-impact running to a user with a declared active bone fracture).

OUTPUT FORMAT SPECIFICATION:
- If the plan is safe, helpful, and reasonable, your output MUST be exactly the string: COMPLIANCE PASSED
- If there is a clear, dangerous physical hazard, your output MUST begin with: CRITICAL REJECTION followed by a clear, short description of the direct hazard.
"""

# =========================================================
# 4. Agent Prompts and Executors Setup (Mapped to specialist_flash_model)
# =========================================================

trainer_prompt_template = ChatPromptTemplate.from_messages([
    ("system", TRAINER_PROMPT),
    ("human", "User Profile: {profile}\nUser Request: {user_message}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
trainer_agent = create_tool_calling_agent(specialist_flash_model, [search_youtube_videos], trainer_prompt_template)
trainer_executor = AgentExecutor(agent=trainer_agent, tools=[search_youtube_videos], verbose=True, handle_parsing_errors=True)

yogi_prompt_template = ChatPromptTemplate.from_messages([
    ("system", YOGI_PROMPT),
    ("human", "User Profile: {profile}\nTrainer Context: {workout}\nUser Request: {user_message}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
yogi_agent = create_tool_calling_agent(specialist_flash_model, [search_youtube_videos], yogi_prompt_template)
yogi_executor = AgentExecutor(agent=yogi_agent, tools=[search_youtube_videos], verbose=True, handle_parsing_errors=True)

dietitian_prompt_template = ChatPromptTemplate.from_messages([
    ("system", DIETITIAN_PROMPT),
    ("human", "User Profile: {profile}\nActivity Context: {workload}\nUser Request: {user_message}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
dietitian_agent = create_tool_calling_agent(specialist_flash_model, [search_and_scrape_recipe], dietitian_prompt_template)
dietitian_executor = AgentExecutor(agent=dietitian_agent, tools=[search_and_scrape_recipe], verbose=True, handle_parsing_errors=True)

safety_prompt_template = ChatPromptTemplate.from_messages([
    ("system", SAFETY_PROMPT),
    ("human", "User Profile: {profile}\n\nGenerated Response:\n{plan}")
])

# =========================================================
# 5. Agent Execution Interfaces
# =========================================================

def run_trainer_agent(user_profile: str, user_message: str) -> str:
    response = trainer_executor.invoke({"profile": user_profile, "user_message": user_message})
    return parse_llm_output(response["output"])

def run_yogi_agent(user_profile: str, user_message: str, workout_plan: str) -> str:
    response = yogi_executor.invoke({"profile": user_profile, "user_message": user_message, "workout": workout_plan})
    return parse_llm_output(response["output"])

def run_dietitian_agent(user_profile: str, user_message: str, workload: str) -> str:
    response = dietitian_executor.invoke({"profile": user_profile, "user_message": user_message, "workload": workload})
    return parse_llm_output(response["output"])

def run_safety_agent(user_profile: str, complete_plan: str) -> str:
    # Uses analytical_pro_model for deep system evaluation
    chain = safety_prompt_template | analytical_pro_model
    response = chain.invoke({"profile": user_profile, "plan": complete_plan})
    return parse_llm_output(response.content)