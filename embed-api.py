from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import spacy

app = FastAPI(title="Word Embedding API", version="1.0")

# 加载 spaCy 模型
nlp = spacy.load("en_core_web_md")

# 定义返回格式
class EmbedResponse(BaseModel):
    word: str
    dim: int
    has_vector: bool
    vector: list[float]

@app.get("/embedding", response_model=EmbedResponse)
def get_embedding(word: str = Query(..., min_length=1, max_length=64)):
    """返回单词的词向量"""
    doc = nlp(word)
    token = doc[0]
    if not token.has_vector:
        raise HTTPException(status_code=404, detail=f"No vector found for '{word}'")
    vec = token.vector.tolist()
    return {"word": token.text, "dim": len(vec), "has_vector": True, "vector": vec}
