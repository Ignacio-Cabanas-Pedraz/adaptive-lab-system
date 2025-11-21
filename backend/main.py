"""
FastAPI Backend for Laboratory Procedure Template Management
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import procedures

app = FastAPI(
    title="Lab Procedure Template API",
    description="API for creating and managing laboratory procedure templates",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(procedures.router, prefix="/api")

@app.get("/")
async def root():
    return {
        "message": "Lab Procedure Template API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI server on http://localhost:8000")
    print("📚 API docs available at http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
