# LLM Evaluation & Monitoring Platform

End-to-end evaluation and monitoring platform for LLM deployments using open-source tools.

## Features

- **LLM Evaluation** — Evaluate any Ollama-hosted model across 6 metrics: hallucination, quality, relevance, factual consistency, NLI-based hallucination, and token cost
- **Toxicity Detection** — Keyword-based (4 categories: insult, toxicity, profanity, threat) with ML fallback and detox pattern filtering
- **Hallucination Scoring** — Cosine similarity, sentence-level NLI, and factual consistency against expected outputs
- **Real-time Monitoring** — 11 custom Prometheus metrics, Grafana dashboard (13 panels), GPU utilization tracking
- **Alerting** — Threshold-based alerts for latency, hallucination, and cost violations
- **Streamlit Dashboard** — Interactive UI for evaluations, history, toxicity/hallucination reports, and quick ad-hoc prompt evaluation
- **REST API** — Full CRUD for prompts, evaluations, models, and alerts
- **MLflow Integration** — Automatic experiment tracking with graceful fallback

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI (Python 3.12) |
| Database | PostgreSQL 16 |
| LLM Serving | Ollama (tinyllama, extensible to any model) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| ML Tracking | MLflow |
| Monitoring | Prometheus + Grafana |
| Caching | Redis |
| Container | Docker + Docker Compose |
| Orchestration | Kubernetes (minikube) |
| CI/CD | GitHub Actions (lint → test → security → docker build) |
| Frontend | Streamlit |

## Quick Start (Docker Compose)

```bash
# 1. Clone the repo
git clone https://github.com/dineshtolani/LLM-evaluaation-platform.git
cd LLM-evaluaation-platform

# 2. Start all services
docker compose up -d --build

# 3. Access the UIs
echo "http://localhost:8000/docs     - API docs"
echo "http://localhost:8501          - Streamlit dashboard"
echo "http://localhost:3000          - Grafana (admin/admin)"
echo "http://localhost:9090          - Prometheus"

# 4. Run a quick evaluation
curl -X POST http://localhost:8000/api/prompts \
  -H "Content-Type: application/json" \
  -d '{"name":"test","content":"What is the capital of France?","category":"qa"}'
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |
| POST | `/api/prompts` | Create a prompt |
| GET | `/api/prompts` | List prompts |
| POST | `/api/evaluations` | Run an evaluation |
| GET | `/api/evaluations` | List evaluations |
| GET | `/api/evaluations/stats` | Aggregate statistics |
| POST | `/api/evaluations/batch` | Batch evaluations |
| GET | `/api/evaluations/toxicity-report` | Toxicity report |
| GET | `/api/evaluations/hallucination-report` | Hallucination report |
| POST | `/api/models` | Register a model |
| GET | `/api/models` | List models |
| POST | `/api/alerts` | Create alert rule |

## Evaluation Metrics

| Metric | Range | Method |
|--------|-------|--------|
| Hallucination Score | 0–1 (lower=better) | Cosine similarity (embeddings) |
| NLI Hallucination | 0–1 (lower=better) | Sentence-level contradiction detection |
| Quality Score | 0–1 (higher=better) | Heuristic: length, diversity, structure |
| Relevance Score | 0–1 (higher=better) | Semantic similarity (embeddings) |
| Factual Consistency | 0–1 (higher=better) | Embedding match vs expected output |
| Toxicity Score | 0–1 (lower=better) | Keyword patterns + ML fallback |
| Token Cost | USD | Configurable per-model rates |

## Kubernetes Deployment

```bash
# Start minikube
minikube start --driver=docker --cpus=4 --memory=8g

# Enable ingress
minikube addons enable ingress

# Deploy
kubectl apply -k k8s/

# Access via port-forward
kubectl port-forward -n llm-eval svc/llm-eval-dashboard 8501:8501
kubectl port-forward -n llm-eval svc/llm-eval-app 8000:8000
kubectl port-forward -n llm-eval svc/llm-eval-grafana 3000:3000

# Or use ingress (add to /etc/hosts)
echo "$(minikube ip) app.llm-eval.local dashboard.llm-eval.local grafana.llm-eval.local prometheus.llm-eval.local" | sudo tee -a /etc/hosts
```

## GPU Support

The platform automatically detects NVIDIA GPUs via `nvidia-smi`. With an RTX A500 (4GB VRAM, CUDA 8.6), models up to 3B params run efficiently. Falls back gracefully to CPU when no GPU is available.

## Monitoring

Prometheus scrapes `/metrics` on the app (port 8000). The Grafana dashboard auto-provisions 13 panels:

- Evaluation rate, latency (P50/P95/P99), hallucination score, toxicity score
- Quality score, token usage, GPU utilization, GPU memory
- Error rate, stat cards (total evaluations, toxic responses, avg latency, alerts)

## Project Structure

```
├── app/
│   ├── evaluation/       # Ollama client, metrics, toxicity
│   ├── models/           # SQLAlchemy models
│   ├── routers/          # FastAPI route handlers
│   ├── schemas/          # Pydantic schemas
│   ├── services/         # MLflow, alerting
│   ├── monitoring/       # Prometheus metrics
│   ├── main.py           # FastAPI entry point
│   └── database.py       # DB session management
├── dashboard/
│   └── app.py            # Streamlit dashboard
├── k8s/                  # Kubernetes manifests
├── grafana/              # Grafana provisioning
├── monitoring/           # Prometheus config
├── tests/                # 86 integration/unit tests
├── docker-compose.yml    # Full stack
├── Dockerfile            # Multi-stage build
└── .github/workflows/    # CI/CD pipeline
```

## License

MIT
