import os
from main import API

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:API", host="0.0.0.0", port=port, reload=False)
