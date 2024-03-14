# Execute from ./LSD/benchmark/xlmroberta



# For Nvidia RTX2080:

# Create the docker
docker build -f dockerfiles/rtx2080/Dockerfile -t xlmroberta .


# For Nvidia RTX3090: (there are some torch incompatibilities)

# Create the docker
docker build -f dockerfiles/rtx3090/Dockerfile -t xlmroberta .



# In both cases, you can run the docker in all available GPUs as:
# UPDATE: the model is not prepared for paralel cumputing, so please, use the bottom command
docker run -it --gpus all xlmroberta

# In both cases, if you want to run the docker in an specific device (0, 1, ..., n):
docker run -it --gpus '"device=0"' xlmroberta
