import json
import os
import random


def process_jsons(train_split=70, eval_split=20, test_split=10):
    train_output = ''
    eval_output = ''
    test_output = ''

    train_counter = 0
    eval_counter = 0
    test_counter = 0

    # Replace 'your_file.json' with the actual path to your JSON file
    map_file_path = 'mapping.json'

    # Read JSON data from file
    with open(map_file_path, 'r') as file:
        data = json.load(file)

    # Convert JSON data to tuple
    subject_map = tuple(data.items())

    # Replace 'your_file.json' with the actual path to your JSON file
    directory = 'spanish'

    # Iterate through the directory tree
    for root, dirs, files in os.walk(directory):
        # Check if the directory name is 'annotations'
        if os.path.basename(root) == 'annotations':
            # Iterate through JSON files in 'annotations' folder
            for filename in files:
                if filename.endswith('.json'):
                    json_path = os.path.join(root, filename)
                    # Process JSON file here
                    print("Processing JSON file:", json_path)
                    # Example: Read JSON data from file
                    with open(json_path, 'r') as json_file:
                        random_number = random.randint(1, 100)

                        json_data = json.load(json_file)
                        all_texts = [entry["text"] for entry in json_data["form"]]
                        all_labels = [entry["label"] for entry in json_data["form"]]
                        text_list = list(zip(all_texts, all_labels))

                        for (item, label) in text_list:
                            words = item.split(" ")
                            tokens = []
                            ner_tags = []
                            match = 0
                            for word in words:
                                word = word.replace('"', '###TEMP_REPLACE###')
                                tokens.append(word)
                                for subject in subject_map:
                                    if label == subject[0]:
                                        ner_tags.append(str(subject[1]))
                                        match = 1

                                if match == 0:
                                    ner_tags.append("0")

                            if random_number <= 70:
                                train_output += "{'id': '" + str(train_counter) + ", 'tokens': " + str(
                                    tokens) + ", 'ner_tags': " + str(
                                    ner_tags) + "}\n"

                                train_counter += 1

                            elif random_number <= 90:
                                eval_output += "{'id': '" + str(eval_counter) + "', 'tokens': " + str(
                                    tokens) + ", 'ner_tags': " + str(
                                    ner_tags) + "}\n"

                                eval_counter += 1

                            else:
                                test_output += "{'id': '" + str(eval_counter) + "', 'tokens': " + str(
                                    tokens) + ", 'ner_tags': " + str(
                                    ner_tags) + "}\n"

                                test_counter += 1

    train_output = train_output.replace("'", '"').replace("###TEMP_REPLACE###", "'")
    eval_output = eval_output.replace("'", '"').replace("###TEMP_REPLACE###", "'")
    test_output = test_output.replace("'", '"').replace("###TEMP_REPLACE###", "'")

    return train_output, eval_output, test_output


def export_dataset(text, output_filename):
    os.chdir(current_directory)
    if not os.path.exists("output_folder"):
        os.makedirs("output_folder")

    output_file_path = os.path.join("output", output_filename)
    with open(output_file_path, 'w', encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    current_directory = os.getcwd()

    (train_output, eval_output, test_output) = process_jsons()

    export_dataset(train_output, "train.json")
    export_dataset(eval_output, "eval.json")
    export_dataset(test_output, "test.json")
