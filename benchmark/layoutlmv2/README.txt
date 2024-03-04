# Execute from ./LSD/benchmark/layoutlmv2



# For Nvidia RTX2080:

# Create the docker
docker build -f Dockerfile -t layoutlmv2 .


# Run the docker in all available GPUs as:
docker run -it --gpus all layoutlmv2
