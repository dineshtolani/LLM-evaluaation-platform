FROM python:3.14-slim AS builder

WORKDIR /build
RUN pip install --no-cache-dir --user streamlit plotly pandas httpx

FROM python:3.14-slim AS production

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

WORKDIR /app
COPY app.py .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/health')"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
