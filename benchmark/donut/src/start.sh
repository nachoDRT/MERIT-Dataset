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

# python src/train.py \
#     --debug False \
#     --dataset_name naver-clova-ix/cord-v2 \
#     --dataset_subset cord-v2 \

# python src/inference.py \
#     --dataset de-Rodrigo/merit \
#     --subset es-render-seq \
#     --model donut_es-render-seq_MERIT-paper \


# python src/inference.py \
#     --dataset dvgodoy/rvl_cdip_mini \
#     --subset rvl_cdip_mini \
#     --model donut_rvl_cdip_mini_MERIT-paper \

python src/inference.py \
    --dataset naver-clova-ix/cord-v2 \
    --subset cord-v2 \
    --model donut_cord-v2_MERIT-paper \
