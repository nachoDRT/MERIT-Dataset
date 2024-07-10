FROM python:3.8.5-slim

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

# Copy the rest of the project
COPY config/ /app/config
COPY cord-target/ /app/cord-target
COPY data/ /app/data
COPY output/ /app/output
COPY src/ /app/src

# Run the server
RUN chmod +x /app/src/start.sh
CMD ["bash", "/app/src/start.sh"]
