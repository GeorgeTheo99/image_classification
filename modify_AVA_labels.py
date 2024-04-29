import json
import numpy as np
import clip_functions.clip_funcs as clip

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

def process_images_and_adjust_labels(json_filepath, image_dir, output_filepath):

    classifier = clip.ImageClassifier()

    with open(json_filepath, 'r') as file:
        data = json.load(file)

    processed_data = []
    x = 1
    for entry in data:
        if x > 20:
            break
        try:
            image_path = f"{image_dir}/{entry['image_id']}.jpg"
            clip_ratings = classifier.get_photo_quality(image_path)
            updated_entry = adjust_label_distribution(entry, clip_ratings)
            processed_data.append(updated_entry)
        except:
            continue
        x+=1

    with open(output_filepath, 'w') as file:
        json.dump(processed_data, file, indent=4)

json_filepath = '/home/georgetheodosopoulos/CVII_project/image-quality-assessment-master_MINE/data/AVA/ava_labels_train.json'
image_dir = '/home/georgetheodosopoulos/CVII_project/image-quality-assessment-master_MINE/training_images/AVA_images/images'
output_filepath = '/home/georgetheodosopoulos/CVII_project/image-quality-assessment-master_MINE/data/AVA/ava_labels_train_CLIP.json'
output_filepath = '/home/georgetheodosopoulos/CVII_project/image-quality-assessment-master_MINE/data/AVA/ava_labels_train_CLIP_TEST.json'
process_images_and_adjust_labels(json_filepath, image_dir, output_filepath)