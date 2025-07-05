chmod +x src/format_dataset.py
chmod +x src/train.py

# python src/format_dataset.py \
#     --test_data_folder True \
#     --gather_train_val_data_from "app/data/train-val/" \
#     --gather_test_data_from "app/data/test/"
# # find . | sed -e 's;[^/]*/;|____;g;s;____|; |;g'

python src/train.py \
    --load_from_hub \
    --dataset_path de-Rodrigo/merit \
    --training_dataset_subset es-digital-token-class \
    --testing_dataset_subset es-render-token-class
