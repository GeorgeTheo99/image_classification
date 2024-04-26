import subprocess
import os
import clip_functions.clip_funcs as clip

def run_nima(image_path, classify_type='aesthetic'):
    if classify_type == 'aesthetic':
        weights_path = os.path.join(os.getcwd(), "image-quality-assessment-master_stable", "models", "MobileNet", "weights_mobilenet_aesthetic_0.07.hdf5")
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
        return result
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)

photo_path = os.path.join(os.getcwd(), "my_test_images", "DJI_0421.jpg")

run_nima(photo_path, 'aesthetic')
classifier = clip.ImageClassifier()
labels = ["great picture", "average picture", "subpar picture"]
classifier.get_photo_quality(photo_path)
print()
classifier.get_photo_category(photo_path)
