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

from database import initialize_database, save_user_profile

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

@app.post("/api/chat")
def generate_wellness_plan(payload: ChatModel):
    if not payload.user_message or not payload.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message cannot be empty.")

    # NOTE: routing decisions (wellness vs. greeting vs. off-topic vs. unsafe,
    # and which specialist to wake up) all happen in ONE place now:
    # orchestrator.py's intent_analyzer_node. This endpoint doesn't need its
    # own keyword filtering or profile pre-checks anymore — the graph handles
    # missing profiles gracefully too (see handle_no_profile_node).
    try:
        final_markdown_plan = execute_wellness_orchestration(
            user_id=payload.user_id,
            user_message=payload.user_message
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
    port = int(os.environ.get("PORT", 8000))
    is_local = os.environ.get("ENV", "local") == "local"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=is_local)
