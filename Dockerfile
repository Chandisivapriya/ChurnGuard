FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir ".[api]"
COPY models ./models
ENV CHURNGUARD_MODEL=models/churn_model.joblib
EXPOSE 8000
CMD ["uvicorn", "churnguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
