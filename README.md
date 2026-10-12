## Generating the lockfile:
    pip-compile requirements.in --output-file=requirements.txt

## Installing dependencies according to requirements.txt
    python -m pip install -r requirements.txt

## Docker command for quadrant db:
    docker run -p 6333:6333 -v C:\pd\docusense-eval-rag\qdrant_data:/qdrant/storage qdrant/qdrant

## Running the API server using uvicorn:
    uvicorn main:app --reload --port 8000

## Swagger UI:
    http://localhost:8000/docs
