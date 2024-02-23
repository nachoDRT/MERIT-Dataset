FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Install Python 3.8
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && python3 -m pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create the working directory
WORKDIR /app

# Copy and install requirements
COPY dockerfiles/rtx2080/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Execute the training
RUN chmod +x run.sh
CMD ["bash", "run.sh"]