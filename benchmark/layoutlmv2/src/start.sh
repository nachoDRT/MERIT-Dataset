# chmod +x src/format_dataset.py
chmod +x src/train.py

# python src/format_dataset.py \
#     --test_data_folder True

# transformers-cli login
# find . | sed -e 's;[^/]*/;|____;g;s;____|; |;g'

python src/train.py \
    --load_from_hub \
    --dataset_path de-Rodrigo/merit \
    --training_dataset_subset en-digital-token-class \
    --testing_dataset_subset en-render-token-class