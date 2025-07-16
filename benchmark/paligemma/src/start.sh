chmod +x src/train.py
chmod +x src/inference.py


# python src/train.py \
#     --dataset_name de-Rodrigo/merit \
#     --dataset_subset es-digital-seq \

# python src/train.py \
#     --dataset_name dvgodoy/rvl_cdip_mini \
#     --dataset_subset rvl_cdip_mini \

# python src/train.py \
#     --dataset_name naver-clova-ix/cord-v2 \
#     --dataset_subset cord-v2 \


# python src/inference.py \
#     --dataset_name de-Rodrigo/merit \
#     --subset_name es-render-seq \
#     --paligemma_model_version de-Rodrigo/paligemma-merit \
#     --subfolder paligemma_es-digital-seq
#     # --paligemma_model_version nielsr/paligemma-cord-demo

python src/inference.py \
    --dataset_name dvgodoy/rvl_cdip_mini \
    --subset_name rvl_cdip_mini \
    --paligemma_model_version de-Rodrigo/paligemma-merit \
    --subfolder paligemma_cord-v2
    