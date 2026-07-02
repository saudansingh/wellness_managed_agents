import traceback
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Injects environment values into runtime memory before local imports execute
load_dotenv()  

from orchestrator import execute_wellness_orchestration
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# Import database initialization and CRUD operations
from database import initialize_database, save_user_profile, get_user_profile_string

# =========================================================
# 1. Initialize FastAPI Application & Lifespan Architecture
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Events
    print("🚀 Starting API Server and initializing cloud databases...")
    initialize_database()
    yield
    # Shutdown Events (Can be left blank or used to close pool connections)
    print("🛑 Shutting down API Server cleanly...")

app = FastAPI(
    title="Managed Wellness Multi-Agent Platform API",
    description="Enterprise backend powering orchestrated fitness, yoga, and nutrition generation.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 2. Define Pydantic Input Data Models
# =========================================================
class OnboardUserRequest(BaseModel):
    userId: str = Field(..., description="Unique identifier matching frontend state.")
    age: int = Field(..., ge=13, le=100)
    weight: float = Field(..., gt=30.0, description="Weight tracking parameter.")
    injuries: Optional[str] = Field("None")
    goals: Optional[str] = Field("", description="Primary goals / restrictions mapped from onboarding.")

class ChatModel(BaseModel):
    user_id: str = Field(..., description="The registered User ID.")
    user_message: str = Field(..., description="The direct question or prompt from the user chat bar.")

# =========================================================
# 3. Define API Endpoints
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
            diet_restrictions=payload.goals  
        )
        return {
            "success": True,
            "message": f"Successfully registered/updated profile for user profile: {payload.userId}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error during user onboarding: {str(e)}")

@app.post("/api/chat")
def generate_wellness_plan(payload: ChatModel):
    profile_check = get_user_profile_string(payload.user_id)
    if "No profile found" in profile_check:
        raise HTTPException(
            status_code=404, 
            detail=f"User ID '{payload.user_id}' has not been onboarded yet. Please complete setup form first."
        )
        
    try:
        current_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_SEARCH_API_KEY") or "NOT_FOUND"
        key_trace = current_key[:15] if len(current_key) > 15 else "INVALID_KEY_LENGTH"
        print(f"\n DEBUG: The API Key being used starts with: {key_trace}...\n")
        
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
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)