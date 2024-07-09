# Execute from ./LSD/benchmark/layoutlmv2



# For Nvidia RTX2080:

# Create the docker
docker build -f dockerfiles/rtx2080/Dockerfile -t layoutlmv2 .


# For Nvidia RTX3090:

# Create the docker
docker build -f dockerfiles/rtx3090/Dockerfile -t layoutlmv2 .


# For Nvidia RTX4090:

# Create the docker
docker build -f dockerfiles/rtx4090/Dockerfile -t layoutlmv2 .


# Run the docker in all available GPUs as:
docker run -it --gpus all layoutlmv2

# UPDATE, the model is not prepared for paralel computing, so please, use the following command
docker run -it --gpus '"device=0"' layoutlmv2
