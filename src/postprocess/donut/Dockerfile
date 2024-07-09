FROM python:3.8.5-slim

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
# Copy the rest of the project
COPY src/ /app/src
COPY cord-target/ /app/cord-target
COPY config/ /app/config

# Run the server
CMD ["python", "/app/src/dataset_to_hf.py"]
