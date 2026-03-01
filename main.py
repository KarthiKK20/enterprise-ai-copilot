from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import ask_question
from tools import calculator_tool
import re
import time

def is_math_expression(q: str) -> bool:
    """
    Detects if the query is a pure arithmetic expression.
    Allows numbers, spaces, and basic math operators.
    """
    return bool(re.fullmatch(r"[0-9+\-*/(). ]+", q.strip()))

app = FastAPI()

class Query(BaseModel):
    question: str
    session_id: str

@app.post("/ask")
def ask(query: Query):
    q = query.question.strip()
    session_id = query.session_id

    if is_math_expression(q):
        result = calculator_tool(q)
        return {"response": result}

    response = ask_question(q, session_id)
    return {"response": response}