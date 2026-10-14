## Starting python venv:
    source venv/Scripts/activate

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

## cURL to check if a model is accessible through Groq:
    check this https://console.groq.com/docs/overview
    ```    
    curl -X POST https://api.groq.com/openai/v1/responses \
    -H "Authorization: Bearer $GROQ_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "openai/gpt-oss-20b",
        "input": "Explain the importance of fast language models"
    }'
    ```

## Setting up frontend repo:
    npm create vite@latest frontend -- --template react-ts
