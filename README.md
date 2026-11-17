## Starting python venv:
    source venv/Scripts/activate

## Generating the lockfile:
    pip-compile requirements.in --output-file=requirements-windows.txt

## Installing dependencies according to requirements-windows.txt
    python -m pip install -r requirements-windows.txt

## Docker command for quadrant db:
    docker run -p 6333:6333 -v C:\pd\docusense-eval-rag\qdrant_data:/qdrant/storage qdrant/qdrant

## web client to access quadrant db data:
    http://localhost:6333/dashboard#/console

## Running the API server using uvicorn:
    uvicorn main:app --reload --port 8000

## Swagger UI:
    http://localhost:8000/docs

## Guest and account documents:
    Open http://localhost:5173 to continue as a guest, upload PDFs, and query or delete them.
    The guest cookie is HttpOnly and lasts 30 days. Create an account or log in to claim PDFs from that browser's guest session; account documents then follow the user across browsers.
    The login cookie is revocable and lasts seven days. Logging out starts a fresh guest session.
    Both cookies require the frontend and backend to use localhost (not mixed localhost/127.0.0.1).

    The API exposes POST /api/v1/auth/register, /login, /logout, GET /api/v1/auth/me, GET/POST /api/v1/documents, and DELETE /api/v1/documents/{document_id}.
    Accounts, login sessions, and document ownership persist through SQLAlchemy ORM in PostgreSQL (the postgres_data Docker volume).
    Qdrant stores session_id or user_id on the vectors, and uploaded PDFs remain in backend/data/uploads.
    PostgreSQL is available locally on 127.0.0.1:5433; set POSTGRES_USER, POSTGRES_PASSWORD, and POSTGRES_DB in the root .env before first startup. The default password is for local development only.
    Changing these settings later does not change credentials in an existing Postgres volume.
    The previous backend/data/app.db is not deleted; it currently contains no account or document records, so there is no existing data to migrate.
    Existing guest PDFs stored under backend/data/uploads/<guest-id>/ are claimed on login; older vectors with no session_id cannot be assigned and need to be re-uploaded.

    This is local email/password authentication, not an identity provider. 
    
## Things to take care when deploying:
    - Use HTTPS and set COOKIE_SECURE=true when deploying over HTTPS; review CSRF protection, rate limiting, email verification, and password recovery before exposing this app publicly.
    - Protect the backend/data directory and Qdrant storage with appropriate filesystem permissions.
    - Do not publish Qdrant's port in a public deployment; direct access bypasses the app's auth.
    - Take care of the pdfs being saved in the backend folder itself

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
    (use -d flag if detached mode is required)
    docker compose up
    docker compose up --watch (generally for dev mode, I want this)
    docker compose up --build
    docker compose up --build --watch

## stoping:
    docker compose stop (to stop while keeping the containers)
    docker compose down (removes the container and Compose network)

## to copy files from host to container when changes are made in the frontend:
    docker compose cp ./frontend/src/. frontend:/app/src/

## if package.json changes (also triggers npm install etc):
    docker compose up -d --build --no-deps frontend
