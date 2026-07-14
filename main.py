import traceback
import os
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



import re

@app.post("/api/chat")
def generate_wellness_plan(payload: ChatModel):
    # 1. Reject empty space instantly
    if not payload.user_message or not payload.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message cannot be empty.")

    cleaned_message = payload.user_message.strip()

    # =========================================================
    # 🚨 TOP-LEVEL FAST INTERCEPTOR (Bypasses Orchestrator Entirely)
    # =========================================================
    # Normalize string: remove punctuation, force lowercase
    normalized_msg = re.sub(r'[^\w\s]', '', cleaned_message.lower()).strip()
    
    # Exact greeting triggers
    casual_tokens = {"hi", "hello", "hey", "sup", "yo", "hola"}
    
    if normalized_msg in casual_tokens:
        print("⚡ [CRITICAL SPEED BYPASS] Returning hardcoded greeting in 1ms.")
        return {
            "success": True,
            "user_id": payload.user_id,
            "plan_markdown": "Hey there! 👋 I'm your wellness assistant. Are we planning a workout, running through a yoga session, or sorting out your nutrition adjustments today?"
        }
    # =========================================================

    # 2. Standard Agent Graph Path (Only triggered for actual plans)
    print("🧬 [Graph Route] Specialized plan requested. Invoking multi-agent workflow...")
    
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


# =========================================================
# Local Execution Entry Point
# =========================================================
if __name__ == "__main__":
    # Cloud Run injects PORT (usually 8080) and requires binding to 0.0.0.0.
    # reload=True is dev-only — never use it in the deployed container.
    port = int(os.environ.get("PORT", 8000))
    is_local = os.environ.get("ENV", "local") == "local"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=is_local)
