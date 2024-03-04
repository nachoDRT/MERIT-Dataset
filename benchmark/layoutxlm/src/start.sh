chmod +x src/format_dataset.py
chmod +x src/train.py

python src/format_dataset.py
# find . | sed -e 's;[^/]*/;|____;g;s;____|; |;g'
python src/train.py
