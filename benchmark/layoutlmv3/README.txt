# Execute from ./LSD/benchmark/layoutlmv3



# For Nvidia RTX2080:

# Create the docker
docker build -f dockerfiles/rtx2080/Dockerfile -t layoutlmv3 .

# For Nvidia RTX3090:

# Create the docker
docker build -f dockerfiles/rtx3090/Dockerfile -t layoutlmv3 .


# Run the docker in all available GPUs as:
docker run -it --gpus all layoutlmv3

# UPDATE, the model is not prepared for paralel computing, so please, use the following command
docker run -it --gpus '"device=0"' layoutlmv3
