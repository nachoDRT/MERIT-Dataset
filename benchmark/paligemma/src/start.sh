chmod +x src/train.py
chmod +x src/inference.py


python src/train.py \
    --dataset_name de-Rodrigo/merit \
    --dataset_subset es-digital-seq \


# python src/inference.py \
#     --dataset_name de-Rodrigo/merit \
#     --subset_name en-digital-seq \
#     --paligemma_model_version nielsr/paligemma-cord-demo