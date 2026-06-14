FROM python:3.14-slim

WORKDIR /app

RUN pip install streamlit plotly pandas httpx

COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
