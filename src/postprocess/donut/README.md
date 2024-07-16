Execute from `./src/postprocess/donut`
### Create the docker :whale:
```bash
docker build -f Dockerfile -t hf_dataset .
```
### Run the docker in your GPU :boom:
```bash
docker run -it -v /host_path_to_save_output_data:/app/merit-dataset-sequence-format hf_dataset
```

Execute from `./src/postprocess/donut/output` to remove a protected `partition` (train/validation/test)
```bash
sudo rm -rf partition
```