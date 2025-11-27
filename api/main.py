from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="CX Synth PoC API")

class Health(BaseModel):
    status: str

@app.get("/health", response_model=Health)
async def health():
    return {"status": "ok"}
