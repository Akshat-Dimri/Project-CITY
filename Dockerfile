FROM python:3.11-slim

# Install Node.js 20
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY nlp_pipeline/requirements.txt ./nlp_pipeline/requirements.txt
RUN pip install --no-cache-dir -r nlp_pipeline/requirements.txt

# Install Node dependencies
COPY backend/package*.json ./backend/
RUN cd backend && npm install

# Copy everything else
COPY . .

WORKDIR /app/backend

EXPOSE 5500
CMD ["node", "server.js"]