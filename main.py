import subprocess
import os
import clip_functions.clip_funcs as clip
import json

def parse_result(result):
    json_part = '\n'.join(result.stdout.split('\n')[2:])
    data = json.loads(json_part)
    first_result = data[0]
    image_id = first_result["image_id"]
    mean_score_prediction = first_result["mean_score_prediction"]
    return (image_id, mean_score_prediction)

def run_nima(image_path, classify_type='aesthetic',clip_data=False):
    if classify_type == 'aesthetic':
        if not clip_data:
            weights_path = os.path.join(os.getcwd(), "image-quality-assessment-master_stable", "models", "MobileNet", "weights_mobilenet_aesthetic_0.07.hdf5")
        elif clip_data:
            weights_path = os.path.join(os.getcwd(), "image-quality-assessment-master_MINE", "models", "MobileNet", "weights_mobilenet_aesthetic_CLIP_01_0.112.hdf5")

    elif classify_type == 'technical':
        weights_path = os.path.join(os.getcwd(), "image-quality-assessment-master_stable", "models", "MobileNet", "weights_mobilenet_technical_0.11.hdf5")
    
    predict_script_path = os.path.join(os.getcwd(), "image-quality-assessment-master_stable", "predict")

    command = [
        "bash", predict_script_path,
        "--docker-image", "nima-cpu",
        "--base-model-name", "MobileNet",
        "--weights-file", weights_path,
        "--image-source", image_path
    ]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(result.stdout)
        return parse_result(result)
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)


def get_all_stats(photo_path = None):
    if not photo_path:
        photo_path = os.path.join(os.getcwd(), "my_test_images", "DJI_0421.jpg")
    image_stats = {
        photo_path: {
            'nima_technical_result': None,
            'nima_aesthetic_result': None,
            'clip_aesthetic_result': None,
            'clip_classification': None,
        }
    }

    nima_technical_result = run_nima(photo_path, 'technical')
    nima_aesthetic_result = run_nima(photo_path, 'aesthetic')
    nima_clip_aesthetic_result = run_nima(photo_path, 'aesthetic', clip_data=True)

    classifier = clip.ImageClassifier()
    clip_aesthetic_result = classifier.get_photo_quality(photo_path)
    clip_classification = classifier.get_photo_category(photo_path)

    image_stats[photo_path]['nima_technical_result'] = nima_technical_result[1]
    image_stats[photo_path]['nima_aesthetic_result'] = nima_aesthetic_result[1]
    image_stats[photo_path]['nima_clip_aesthetic_result'] = nima_clip_aesthetic_result[1]
    image_stats[photo_path]['clip_aesthetic_result'] = clip_aesthetic_result
    image_stats[photo_path]['clip_classification']   = clip_classification

    print(image_stats)
    return image_stats

if __name__=='__main__':
    get_all_stats()