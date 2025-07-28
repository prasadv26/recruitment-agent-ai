# Official Python base image
FROM python:3.10-slim

WORKDIR /app

# system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY . /app

#Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "recruiting_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
