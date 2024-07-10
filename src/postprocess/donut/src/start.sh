chmod +x src/dataset_to_hf.py
chmod +x src/filter_data.py

python src/filter_data.py \
    --test_data_folder True \
    --language english
# python src/dataset_to_hf.py