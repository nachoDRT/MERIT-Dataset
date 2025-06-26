chmod +x src/train.py
chmod +x src/inference.py


# python src/train.py \
#     --dataset_name de-Rodrigo/merit \
#     --dataset_subset en-digital-seq \

python src/train.py \
    --dataset_name dvgodoy/rvl_cdip_mini \
    --dataset_subset rvl_cdip_mini \


# python src/inference.py \
#     --dataset_name de-Rodrigo/merit \
#     --subset_name en-digital-seq \
#     --paligemma_model_version nielsr/paligemma-cord-demo