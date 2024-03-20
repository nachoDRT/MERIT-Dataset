chmod +x src/format_dataset.py
chmod +x src/train.py

python src/format_dataset.py \
    --test_data_folder True
# transformers-cli login
# find . | sed -e 's;[^/]*/;|____;g;s;____|; |;g'
python src/train.py