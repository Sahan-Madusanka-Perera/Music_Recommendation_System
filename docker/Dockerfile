FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -v \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    libglib2.0-0 libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.address", "0.0.0.0"]