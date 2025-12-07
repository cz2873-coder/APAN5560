# api.py
# Example FastAPI integration for the fine-tuned GPT-2 model.

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.post("/generate_with_llm")
def generate_with_llm(request: TextGenerationRequest):
    prompt = f"Question: {request.start_word}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(inputs["input_ids"], max_new_tokens=request.length)
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    return {"generated_text": text}
