import json
import numpy as np
import pandas as pd
import clip_functions.clip_funcs as clip
import os

def adjust_label_distribution(image_entry, clip_ratings):
    original_labels = np.arange(1, len(image_entry['label']) + 1)
    original_distribution = np.array(image_entry['label'])

    weights = np.array([
        clip_ratings['subpar picture'],   # Index 0
        clip_ratings['subpar picture'],   # Index 1
        clip_ratings['subpar picture'],   # Index 2
        clip_ratings['average picture'],  # Index 3
        clip_ratings['average picture'],  # Index 4
        clip_ratings['average picture'],  # Index 5
        clip_ratings['average picture'],  # Index 6
        clip_ratings['great picture'],    # Index 7
        clip_ratings['great picture'],    # Index 8
        clip_ratings['great picture']     # Index 9
    ])

    weights_normalized = weights / weights.sum() * original_distribution.sum()

    adjusted_distribution = original_distribution * weights_normalized

    adjusted_distribution_normalized = adjusted_distribution / adjusted_distribution.sum() * original_distribution.sum()

    updated_entry = image_entry.copy()
    updated_entry['label'] = adjusted_distribution_normalized.astype(int).tolist()

    return updated_entry


def process_images_and_adjust_labels(json_filepath, image_dir, output_filepath_json, output_filepath_csv, chunk_size=100):
    classifier = clip.ImageClassifier()
    df_columns = ['image_id', 'subpar picture', 'average picture', 'great picture']
    df = pd.DataFrame(columns=df_columns)

    with open(json_filepath, 'r') as file:
        data = json.load(file)

    processed_data = []
    count = 0
    break_count = 0
    for entry in data:
        # if break_count >= 1000:
        #     break
        # break_count += 1
        try:
            image_path = f"{image_dir}/{entry['image_id']}.jpg"
            clip_ratings = classifier.get_photo_quality(image_path)
            updated_entry = adjust_label_distribution(entry, clip_ratings)
            processed_data.append(updated_entry)
            df_row = pd.DataFrame({
                'image_id': [entry['image_id']],
                'subpar picture': [clip_ratings['subpar picture']],
                'average picture': [clip_ratings['average picture']],
                'great picture': [clip_ratings['great picture']]
            })
            if not df.empty:
                df = pd.concat([df, df_row], ignore_index=True)
            else:
                df = df_row
            count += 1

            if count % chunk_size == 0:
                write_header = not os.path.exists(output_filepath_csv)
                df.to_csv(output_filepath_csv, mode='a', header=write_header, index=False)
                df = pd.DataFrame(columns=df_columns)  # Reset the DataFrame

        except Exception as e:
            print(f"Error processing image {entry['image_id']}: {str(e)}")
            continue

    # Write remaining data if any
    if not df.empty:
        df.to_csv(output_filepath_csv, mode='a', header=False, index=False)

    with open(output_filepath_json, 'w') as file:
        json.dump(processed_data, file, indent=4)

json_filepath = '/home/georgetheodosopoulos/CVII_project/image-quality-assessment-master_MINE/data/AVA/ava_labels_test.json'
image_dir = '/home/georgetheodosopoulos/CVII_project/image-quality-assessment-master_MINE/training_images/AVA_images/images'
output_filepath_json = '/home/georgetheodosopoulos/CVII_project/image-quality-assessment-master_MINE/data/AVA/ava_labels_test_CLIP.json'
output_filepath_csv = '/home/georgetheodosopoulos/CVII_project/image-quality-assessment-master_MINE/data/AVA/ava_labels_test_CLIP.csv'

process_images_and_adjust_labels(json_filepath, image_dir, output_filepath_json, output_filepath_csv)
