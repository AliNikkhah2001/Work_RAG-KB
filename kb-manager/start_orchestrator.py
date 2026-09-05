import uvicorn
from kb_manager.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="info")