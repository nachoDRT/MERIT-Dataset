# Nvidia RTX2080/3090/4090
Execute from ./VrDU-Doctor/single/idefics2

Available cards: `RTX2080`, `RTX3090`, `RTX4090`

### Create the docker :whale:
```bash
docker build -f dockerfiles/your_card/Dockerfile -t idefics2 .
```

### Run :boom: or Debug :no_entry_sign::bug: the docker
```bash
docker run -p 5678:5678 -it --gpus all -v "$HOME/.cache/huggingface":/root/.cache/huggingface -v /host_path_to_save_models:/app/models_output --ipc=host idefics2 2>&1 | tee log.txt
```

### Inspect default dataset using a volume
```bash
docker run -v /host_path_to_save_dataset_inspection:/app/dataset_inspection idefics2
```

# Remove *models* protected folder
Execute from ./VrDU-Doctor/single/idefics2
```bash
sudo rm -rf models
```