# Execute from ./LSD/benchmark/layoutlmv3



# For Nvidia RTX2080:

# Create the docker
docker build -f Dockerfile -t layoutxlm .


# Run the docker in all available GPUs as:
docker run -it --gpus all layoutxlm
