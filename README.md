## Starting python venv:
    source venv/Scripts/activate

## Generating the lockfile:
    pip-compile requirements.in --output-file=requirements-windows.txt

## Installing dependencies according to requirements-windows.txt
    python -m pip install -r requirements-windows.txt

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

## Updating the requirements-linux.txt
    ```
    MSYS_NO_PATHCONV=1 docker run --rm \
    -v "C:/pd/docusense-eval-rag/backend:/work" \
    -w /work \
    python:3.13-slim \
    sh -c "pip install --no-cache-dir pip-tools && pip-compile requirements.in --output-file=requirements-linux.txt"
    ```
    Run it before the first Docker Compose build, and again whenever you change requirements.in. It creates a temporary Linux container, installs pip-tools inside it, and runs pip-compile. Because backend is mounted at /work, the resulting requirements-linux.txt is written to your project folder. --rm removes the temporary helper container when it exits; it does not start your app or install anything into the backend container.

## Starting up:
    docker compose up
    docker compose up --build
    docker compose up --build --watch
