from main import get_all_stats
from PIL import Image
import os
import pandas as pd
import re
import json


with open('/home/georgetheodosopoulos/CVII_project/image-quality-assessment-master_MINE/data/AVA/ava_labels_test.json', 'r') as file:
    original_data = json.load(file)

with open('/home/georgetheodosopoulos/CVII_project/image-quality-assessment-master_MINE/data/AVA/ava_labels_test_CLIP.json', 'r') as file:
    modified_data = json.load(file)

df_og = pd.DataFrame(original_data)
df_clip = pd.DataFrame(modified_data)

# Calculate the average rating
def calculate_average_rating(labels):
    ratings = range(1, 11)
    total_ratings = sum(labels)
    weighted_sum = sum(rating * count for rating, count in zip(ratings, labels))
    return weighted_sum / total_ratings if total_ratings != 0 else 0

def extract_data(image_path):
    results = get_all_stats(image_path)
    _, data = next(iter(results.items()))
    def process_level(level_data):
        if level_data:
            return ', '.join(level_data)
        return None

    return {
        'nima_technical_result': data['nima_technical_result'],
        'nima_aesthetic_result': data['nima_aesthetic_result'],
        'nima_clip_aesthetic_result': data['nima_clip_aesthetic_result'],
        'great_picture': data['clip_aesthetic_result']['great picture'],
        'average_picture': data['clip_aesthetic_result']['average picture'],
        'subpar_picture': data['clip_aesthetic_result']['subpar picture'],
        'level_1': process_level(data['clip_classification'].get('level_1', [])),
        'level_2': process_level(data['clip_classification'].get('level_2', [])),
        'level_3': process_level(data['clip_classification'].get('level_3', []))  # Assuming level_3 can also be a key
    }


df_og['average_rating_ava'] = df_og['label'].apply(calculate_average_rating)
df_clip['average_rating_ava_clip'] = df_clip['label'].apply(calculate_average_rating)

df_og = df_og.drop(columns=['label'])
df_clip = df_clip.drop(columns=['label'])

# Ensure that 'image_id' is preserved as a column after merging
df = pd.merge(df_og, df_clip, on='image_id', how='inner')

base_path = "/home/georgetheodosopoulos/CVII_project/image-quality-assessment-master_MINE/training_images/AVA_images/images/"
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(base_path, f"{x}.jpg"))

# Select a sample size of 10,000 images; replaced with 5 for testing
sample_df = df.sample(n=1000, random_state=1)

# Applying extract_data function and avoiding index misalignment
sample_df['stats'] = sample_df['image_path'].apply(extract_data)
stats_results = pd.json_normalize(sample_df['stats'])

# Reset index before concatenation to avoid size issues
df_final = pd.concat([sample_df.reset_index(drop=True), stats_results.reset_index(drop=True)], axis=1)

# Export to CSV, ensure this path is writable
df_final.to_csv('all_stats_output_2.csv', index=False)