import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Import the Gradio Blocks app from app.py
from app import demo

app = FastAPI()

# Mount Gradio under /app
app = demo.mount(app, path="/app")

# Templates for index page (with ads)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
