FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV PYTHONUNBUFFERED=1 LLM_MAX_CONCURRENCY=2 LLM_MAX_QUEUE=16
EXPOSE 8000
CMD ["uvicorn", "enterprise_llm.api:app", "--host", "0.0.0.0", "--port", "8000"]
