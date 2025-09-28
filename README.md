# Word Embedding API

This project is a FastAPI service that returns spaCy word embeddings.

## Run locally
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
uvicorn embed-api:app --reload
