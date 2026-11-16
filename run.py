"""
Entry point to run the Gillo AI Service.
Usage:
    .\.venv\Scripts\python run.py
"""
import os
import sys
import uvicorn
from app.config import settings

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.dirname(__file__))
    print(f"Starting Gillo AI Service from: {project_root}")
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=settings.SERVICE_PORT,
        reload=True,
        reload_dirs=[os.path.join(project_root, "app")]
    )
