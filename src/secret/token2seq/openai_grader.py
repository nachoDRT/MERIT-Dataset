from openai import OpenAI
import numpy as np
from utils import *
import cv2
from PIL import Image
from pathlib import Path
from os.path import join, abspath, dirname
import argparse


def get_output_seq(base64_image, gt, client, list_ft_models: bool = False):

    if list_ft_models:
        list_fine_tunes(client)

    try:
        model = get_model()
    except:
        model = "gpt-4o"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an assistant that extracts grades from students' transcripts of records.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Look at the image and extract:\n"
                            "- The subjects and their grades.\n"
                            "- The level (3 de la eso, 4 de la eso, 1 de bachillerato, or 2 de bachillerato) they correspond to.\n\n"
                            "You must return a SINGLE JSON object in the exact following format:\n\n"
                            "{\n"
                            '  "3_de_la_eso": [  # Or 4_de_la_eso, 1_de_bachillerato, or 2_de_bachillerato as appropriate\n'
                            '    {"subject": "...", "grade": "..."},\n'
                            '    {"subject": "...", "grade": "..."}\n'
                            "  ]\n"
                            "}\n\n"
                            "DO NOT include any additional text, explanations, or comments. "
                            "Use the key '3_de_la_eso', '4_de_la_eso', '1_de_bachillerato', or '2_de_bachillerato' based on what can be inferred from the image."
                            "To help you, you can infer the grades from the Dictionary I give you. Also, you should match every subject area given in this "
                            f"dictionary with a subject extracted from the page: {gt}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
    )

    grades = detect_json(str(response.choices[0]))
    grades = clean_json(grades)

    return grades


def process_dataset(dataset_iterator):

    client = OpenAI()

    for i, sample in enumerate(dataset_iterator):

        print(f"Processing img {i}")
        image, gt = get_sample_data(sample)

        if isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        base64_image = encode_image(image)

        seq = get_output_seq(base64_image, gt, client)

        if img is not None:
            height, width = img.shape[:2]
            new_size = (width // 2, height // 2)
            img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            cv2.imshow(f"Image {i}", img_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def process_local_dataset(directory: str, show_imgs: bool = False):
    client = OpenAI()

    curated_dir = Path(join(dirname(dirname(abspath(__file__))), "curated"))
    curated_dir.mkdir(parents=True, exist_ok=True)

    for json_file in sorted(Path(directory).glob("*.json")):
        print(f"Processing file: {json_file.name}")

        image, gt = get_local_sample_data(json_file)

        if isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image

        base64_image = encode_image(image)
        seq = get_output_seq(base64_image, gt, client)

        if img is not None:
            height, width = img.shape[:2]
            new_size = (width // 2, height // 2)
            img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

            curated_json_path = curated_dir / json_file.name
            curated_image_path = curated_dir / f"{json_file.stem}.png"

            with open(curated_json_path, "w", encoding="utf-8") as f:
                json.dump(seq, f, indent=4)

            cv2.imwrite(str(curated_image_path), img)
            print(f"Saved curated files: {curated_json_path}, {curated_image_path}")

            if show_imgs:
                cv2.imshow(f"Image {json_file.stem}", img_resized)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--school", required=True, type=str)
    parser.add_argument("--show", required=True, type=str2bool, nargs="?", const=True)
    args = parser.parse_args()

    school = args.school
    show_imgs = args.show

    init_apis()

    # dataset_iterator = get_dataset_iterator()
    # process_dataset(dataset_iterator)

    local_files_path = join(dirname(dirname(abspath(__file__))), "data", school)
    process_local_dataset(local_files_path, show_imgs)


if __name__ == "__main__":

    main()
