chmod +x src/train.py
chmod +x src/inference.py

python src/train.py \
    --debug False \
    --dataset de-Rodrigo/merit \
    --subset en-render-seq \
    # --save_initial

# python src/train.py \
#     --debug False \
#     --dataset dvgodoy/rvl_cdip_mini \
#     --subset rvl_cdip_mini \
#     # --save_initial

# python src/inference.py \
#     --dataset de-Rodrigo/merit-secret \
#     --subset all \
#     --model es-digital-rotation-zoom-degradation-seq