chmod +x src/train.py

# python src/train.py \
#     --debug False \
#     --dataset_name de-Rodrigo/merit \
#     --dataset_subset en-digital-seq \
#     --base_subset en-digital-seq \
#     --school_name_subset freefields \
#     --school_name_subset james \
#     --school_name_subset paloalto \
#     --school_name_subset pinnacle \
#     --school_name_subset whitney \


# python src/train.py \
#     --debug False \
#     --dataset_name dvgodoy/rvl_cdip_mini \
#     --dataset_subset rvl_cdip_mini \

python src/inference.py \
    --dataset de-Rodrigo/merit \
    --subset en-render-seq \
    --model donut_en-digital-seq_MERIT-paper \
