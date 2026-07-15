import traceback
import os
import re
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from orchestrator import execute_wellness_orchestration
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from database import initialize_database, save_user_profile, get_user_profile_string

# =========================================================
# 1. Initialize FastAPI Application & Lifespan Architecture
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting API Server and initializing cloud databases...")
    initialize_database()
    yield
    print("🛑 Shutting down API Server cleanly...")

app = FastAPI(
    title="Managed Wellness Multi-Agent Platform API",
    description="Enterprise backend powering orchestrated fitness, yoga, and nutrition generation.",
    version="1.0.0",
    lifespan=lifespan
)

# In production, set FRONTEND_ORIGIN to your deployed frontend's exact URL
# (e.g. "https://your-app.web.app"). Wildcard "*" is fine for local dev only.
ALLOWED_ORIGINS = os.environ.get("FRONTEND_ORIGIN", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 2. Pydantic Input Models
# =========================================================
class OnboardUserRequest(BaseModel):
    userId: str = Field(..., description="Unique identifier matching frontend state.")
    age: int = Field(..., ge=13, le=100)
    weight: float = Field(..., gt=30.0, description="Weight tracking parameter.")
    injuries: Optional[str] = Field("None")
    goals: Optional[str] = Field("", description="Primary goals / focus mapped from onboarding.")

class ChatModel(BaseModel):
    user_id: str = Field(..., description="The registered User ID.")
    user_message: str = Field(..., description="The direct question or prompt from the user chat bar.")

# =========================================================
# 3. API Endpoints
# =========================================================

@app.get("/")
def read_root():
    return {"status": "healthy", "service": "wellness-agent-orchestrator"}

@app.post("/api/save-profile", status_code=201)
def onboard_user(payload: OnboardUserRequest):
    try:
        save_user_profile(
            user_id=payload.userId,
            age=payload.age,
            weight=payload.weight,
            injuries=payload.injuries,
            goals=payload.goals
        )
        return {
            "success": True,
            "message": f"Successfully registered/updated profile for user profile: {payload.userId}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error during user onboarding: {str(e)}")

# @app.post("/api/chat")
# def generate_wellness_plan(payload: ChatModel):
#     # Reject blank/whitespace-only messages up front (previously these were
#     # passed straight through to the LLM router, wasting a call for nothing).
#     if not payload.user_message or not payload.user_message.strip():
#         raise HTTPException(status_code=400, detail="user_message cannot be empty.")

#     # profile_check = get_user_profile_string(payload.user_id) ( i have commented this to paas normal msg direct)
#     # if "No profile found" in profile_check:
#     #     raise HTTPException(
#     #         status_code=404,
#     #         detail=f"User ID '{payload.user_id}' has not been onboarded yet. Please complete setup form first."
#     #     )

#     try:
#         final_markdown_plan = execute_wellness_orchestration(
#             user_id=payload.user_id,
#             user_message=payload.user_message
#         )

#         return {
#             "success": True,
#             "user_id": payload.user_id,
#             "plan_markdown": final_markdown_plan
#         }
#     except Exception as e:
#         traceback.print_exc()
#         raise HTTPException(
#             status_code=500,
#             detail=f"Orchestration failure during agent execution sequence: {str(e)}"
#         )





@app.post("/api/chat")
def generate_wellness_plan(payload: ChatModel):
    if not payload.user_message or not payload.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message cannot be empty.")

    cleaned_message = payload.user_message.strip()
    normalized_msg = cleaned_message.lower()

    # =========================================================
    # 🧠 SMART SEMANTIC ROUTER (Wellness Trigger Keywords)
    # =========================================================
    # If the message contains ANY of these words, it needs a specialist agent.
    wellness_keywords = {
        "workout", "train", "exercise", "gym", "routine", "lift", "cardio",
        "yoga", "stretch", "mobility", "asana", "pain", "stiff", "flexibility",
        "diet", "food", "eat", "recipe", "macro", "calorie", "protein", "carb",
        "veg", "non-veg", "weight", "gain", "lose", "fat", "muscle", "plan"
    }

    # Check if any wellness keyword is present in the user's message
    is_wellness_request = any(word in normalized_msg for word in wellness_keywords)

    if not is_wellness_request:
        print("⚡ [Fast Casual Brain] Handling general conversation directly.")
        
        # Pull the recent history context so it flows like a real 2-way conversation
        from database import get_recent_history, save_chat_message
        from agents import analytical_pro_model
        
        history_context = get_recent_history(payload.user_id, turns=6) or "No prior history"

        # Ask the LLM to act like a natural conversationalist who gently guides the user
        conversational_prompt = f"""You are a friendly, conversational AI Wellness Assistant. 
The user is talking casually with you, responding to a prompt, or asking general questions. 

Guidelines:
1. Talk like a real human in a natural 2-way conversation. Respond directly to their statement.
2. Keep your answer brief, warm, and engaging (1 to 2 sentences max).
3. Do NOT repeat an error message or say "I didn't catch that". 
4. If they do not want to discuss fitness right now, politely acknowledge it or chat with them casually.

Recent Chat History:
{history_context}

User's current message: "{cleaned_message}"
Response:"""

        try:
            # Fast raw LLM call completely bypassing the heavy LangGraph overhead
            bot_reply = analytical_pro_model.invoke(conversational_prompt).content.strip()
            
            # Persist history so the context stays unbroken
            save_chat_message(payload.user_id, "user", cleaned_message)
            save_chat_message(payload.user_id, "assistant", bot_reply)

            return {
                "success": True,
                "user_id": payload.user_id,
                "plan_markdown": bot_reply
            }
        except Exception as e:
            print(f"Fallback error in casual handler: {e}")

    # =========================================================
    # 🧬 ENTERPRISE AGENT GRAPH PATH (Only triggered for plans)
    # =========================================================
    print("🧬 [Graph Route] Wellness intent identified. Invoking specialized agents...")
    
    profile_check = get_user_profile_string(payload.user_id)
    if "No profile found" in profile_check:
        raise HTTPException(
            status_code=404,
            detail=f"User ID '{payload.user_id}' has not been onboarded yet. Please complete setup form first."
        )

    try:
        final_markdown_plan = execute_wellness_orchestration(
            user_id=payload.user_id,
            user_message=cleaned_message
        )

        return {
            "success": True,
            "user_id": payload.user_id,
            "plan_markdown": final_markdown_plan
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Orchestration failure during agent execution sequence: {str(e)}"
        )
