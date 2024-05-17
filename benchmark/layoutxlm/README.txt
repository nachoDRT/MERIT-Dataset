# Execute from ./LSD/benchmark/layoutlxlm


# A. IF YOU WANT TO RUN TRAINING SESSION WITH AN SPECIFIC DATASET 
# For Nvidia RTX2080:

# Create the docker
docker build -f dockerfiles/rtx2080/Dockerfile -t layoutxlm .

# For Nvidia RTX3090:

# Create the docker
docker build -f dockerfiles/rtx3090/Dockerfile -t layoutxlm .


# For Nvidia RTX4090:

# Create the docker
docker build -f dockerfiles/rtx4090/Dockerfile -t layoutxlm .


# Run the docker in all available GPUs as:
docker run -it --gpus all layoutxlm

# UPDATE, the model is not prepared for paralel computing, so please, use the following command
docker run -it --gpus '"device=0"' layoutxlm

____________________________________________________________
# B. IF YOU WANT TO RUN A BRUTEFORCE TRAINING SESSION WITH 'N' SUBSETS COMBINATIONS DATASET

# For Nvidia RTX2080:

# Create the docker
docker build -f dockerfiles/rtx2080_bruteforce/Dockerfile -t layoutxlm_bruteforce .

# For Nvidia RTX3090:

# Create the docker
docker build -f dockerfiles/rtx3090_bruteforce/Dockerfile -t layoutxlm_bruteforce .

# Run the docker in your GPU:
docker run -it --gpus '"device=0"' layoutxlm_bruteforce
