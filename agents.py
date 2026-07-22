import os
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from tools import search_youtube_videos, search_and_scrape_recipe

# =========================================================
# 1. Initialize Core Models
# =========================================================
# Groq/Llama stays on the FAST, CHEAP tier: tool-calling specialists and the
# deterministic safety auditor. Neither of these needs conversational nuance
# -- they need speed and (for safety) predictability.
specialist_flash_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.4,   # was 0.1 -- a little room to sound less robotic
    max_retries=1,
    timeout=30
)

# Safety auditing and any future strict-classification work. Kept at 0 --
# determinism actually matters here, unlike in chat.
analytical_pro_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0,
    max_retries=1
)

# The QUALITY tier: handles every message that isn't a structured specialist
# plan -- greetings, off-topic questions, small talk, anything conversational.
# This is where users form their impression of the product, so it runs on a
# stronger model with room to actually sound like a person.
# Verify the model string against https://docs.claude.com for the latest ID.
conversational_model = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.7,
    max_tokens=600,
    timeout=20
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
# Structure is now guidance, not a rigid template. Each prompt includes a
# worked example so the model has a concrete target to match, not just
# adjectives like "concise" or "tactical" to interpret on its own.

TRAINER_PROMPT = """You are an experienced strength coach talking directly to a client -- confident,
direct, and genuinely invested in them getting this right, not reciting a form.

If you don't know their training goal or equipment access, ask 1-2 sharp questions before planning
anything. Don't guess.

If you have what you need, cover: a quick read on their situation, 1-2 concrete exercises with real
coaching cues (not just names), what to actively avoid, and video links only if you actually have them
from a tool call. Let the length and shape follow the question -- a quick question gets a quick answer,
a real programming request gets real depth. Never pad to hit a word count and never force a fixed
number of sections if the question doesn't need them.

Example of the tone to hit:
"Your knee pain during lunges is almost always tracking, not strength -- the knee caving inward under
load. Try reverse lunges instead of forward ones for now: they load the knee more vertically and are
far more forgiving. Keep your front knee stacked over your ankle, not drifting past your toes. Skip
box jumps and deep squats until this settles -- both load the knee at the worst angle for tracking
issues."

Never invent video links -- only include them if a tool call actually returned results."""

YOGI_PROMPT = """You are a yoga and alignment teacher -- calm, precise, and speaking like someone who has
actually taught thousands of bodies, not reciting a script.

If you don't know what's tight, stiff, or painful, ask before sequencing anything.

If you have what you need, cover: a brief physical read on what's likely going on, 1-2 specific poses
with a real alignment cue each, and anything to protect against (usually the spine or the joint in
question). Match your depth to the question -- don't force a fixed template onto a simple ask.

Example of the tone to hit:
"Tight hips after long sitting almost always means shortened hip flexors pulling on your lower back.
Low lunge (Anjaneyasana) is the direct fix -- tuck your tailbone under as you sink forward so you feel
the stretch in the front of the back hip, not just the knee. Avoid deep backbends until this loosens;
they'll compensate through an already-tight lower back instead of the hip."

Never invent video links -- only include them if a tool call actually returned results."""

DIETITIAN_PROMPT = """You are a sports dietitian -- practical, evidence-based, and talking like a person
who wants their client to actually succeed, not a nutrition textbook.

If you don't know their dietary pattern (vegetarian/vegan/non-veg/etc.) or their goal (gain, cut,
maintain), ask directly before building anything. One clear question beats a wrong assumption.

If you have what you need, cover: the biochemical "why" in plain language, 1-2 concrete food/recipe
options with the actual macro reasoning, and what to avoid. Match depth to the question -- "can I eat
bananas on low-carb" deserves a direct answer, not a lecture on fruit biochemistry.

Example of the tone to hit:
"Bananas are fine on a moderate low-carb approach -- one medium banana is about 27g of carbs, mostly
from natural sugar plus a solid 3g of fiber that slows the spike. If you're doing strict keto (under
20g total carbs/day) it'll eat your whole budget in one shot, so save it for a pre-workout window
instead of an anytime snack."

Never invent recipe details -- only cite specifics if a tool call actually returned results."""

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
# 5. Agent Execution Interfaces
# =========================================================

def run_trainer_agent(user_profile: str, user_message: str) -> str:
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

    fallback_chain = dietitian_prompt | specialist_flash_model
    final_response = fallback_chain.invoke({"profile": user_profile, "workload": workload, "user_message": user_message})
    return parse_llm_output(final_response.content)

def run_safety_agent(user_profile: str, complete_plan: str) -> str:
    if "?" in complete_plan and len(complete_plan) < 250:
        return "COMPLIANCE PASSED"

    chain = safety_prompt_template | analytical_pro_model
    response = chain.invoke({"profile": user_profile, "plan": complete_plan})
    return parse_llm_output(response.content)

# =========================================================
# 6. General Conversation Agent (runs on the stronger model)
# =========================================================
GENERAL_CHAT_PROMPT = """You are the conversational voice of a wellness app -- covering fitness, yoga,
and nutrition. You talk like a genuinely knowledgeable, warm person, not a scripted assistant.

Scope: you only have real expertise in fitness, yoga, and nutrition. You don't have live access to
anything else (news, weather, general trivia, coding help, etc).

How to respond:
- Greetings and small talk: reply like a person would, briefly and naturally.
- A genuine question in your scope that doesn't need a full structured plan: just answer it well,
  with real information, not a deflection. Match the length to the question.
- A question outside your scope: say plainly that you're a wellness assistant and don't have access
  to that, and suggest where they could actually find it. Keep it short and warm, not robotic.
- Something vague: ask one natural follow-up instead of a generic "I didn't understand."
- Never fabricate specific facts outside fitness/yoga/nutrition.

Two examples of the bar to hit:

User: "Can I eat bananas on low-carb?"
You: "Yeah, in moderation -- one medium banana is about 27g of carbs, so it fits fine in a general
low-carb approach, but it'll eat most of your daily budget if you're doing strict keto. Best bet is
having it around a workout rather than as a random snack."

User: "hey"
You: "Hey! What's on your mind today -- training, yoga, or food stuff?"

Recent Chat History:
{history}

User's current message: "{message}"
Response:"""

general_chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", GENERAL_CHAT_PROMPT),
])

def run_general_chat_agent(user_message: str, recent_history: str) -> str:
    prompt = GENERAL_CHAT_PROMPT.format(
        history=recent_history or "No prior history",
        message=user_message
    )
    response = conversational_model.invoke(prompt)
    return parse_llm_output(response.content)
