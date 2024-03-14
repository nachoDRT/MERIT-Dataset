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

RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -q detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html

# Copy and install requirements
COPY dockerfiles/rtx3090/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Execute the training
RUN chmod +x src/start.sh
CMD ["bash", "src/start.sh"]
