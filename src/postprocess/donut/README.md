Execute from ./src/postprocess/donut
### Create the docker :whale:
```bash
docker build -f Dockerfile -t hf_dataset .
```
### Run the docker in your GPU :boom:
```bash
docker run -it hf_dataset
```