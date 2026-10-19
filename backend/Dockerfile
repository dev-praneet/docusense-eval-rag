FROM python:3.13-slim

WORKDIR /app/backend
COPY requirements-linux.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
	pip install -r requirements-linux.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
