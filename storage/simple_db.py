from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import os

_DB_PATH = os.environ.get("CX_DB_PATH", "./data/cx.db")
_engine: Engine | None = None

def get_engine() -> Engine:
    global _engine
    if _engine is None:
        os.makedirs("./data", exist_ok=True)
        _engine = create_engine(f"sqlite:///{_DB_PATH}", future=True)
    return _engine
