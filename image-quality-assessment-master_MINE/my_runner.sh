#!/bin/bash


# bash ./predict --docker-image nima-cpu \
# --base-model-name MobileNet \
# --weights-file $(pwd)/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5 \
# --image-source $(pwd)/src/tests/test_images/DJI_0421.jpg

# bash ./train-local \
# --config-file $(pwd)/models/MobileNet/config_technical_cpu.json \
# --samples-file $(pwd)/data/TID2013/tid_labels_train.json \
# --image-dir $(pwd)/training_images/TID2013_images

# bash ./predict --docker-image nima-cpu \
# --base-model-name MobileNet \
# --weights-file $(pwd)/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5 \
# --image-source $(pwd)/src/tests/test_images/DJI_0421.jpg

# bash ./predict --docker-image nima-cpu \
# --base-model-name MobileNet \
# --weights-file $(pwd)/models/MobileNet/weights_mobilenet_technical_0.11.hdf5 \
# --image-source $(pwd)/src/tests/test_images/DJI_0421.jpg

# bash ./predict --docker-image nima-cpu \
# --base-model-name MobileNet \
# --weights-file $(pwd)/train_jobs/2024_04_26_06_04_36/weights/weights_mobilenet_06_0.137.hdf5 \
# --image-source $(pwd)/src/tests/test_images/DJI_0421.jpg

bash ./train-local \
--config-file $(pwd)/models/MobileNet/config_aesthetic_cpu.json \
--samples-file $(pwd)/data/AVA/ava_labels_train_CLIP.json \
--image-dir $(pwd)/training_images/AVA_images/images

# bash ./predict --docker-image nima-cpu \
# --base-model-name MobileNet \
# --weights-file $(pwd)/train_jobs/2024_04_28_04_02_57/weights/weights_mobilenet_01_0.112.hdf5 \
# --image-source $(pwd)/src/tests/test_images/DJI_0421.jpg


# bash ./predict --docker-image nima-cpu \
# --base-model-name MobileNet \
# --weights-file $(pwd)/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5 \
# --image-source $(pwd)/src/tests/test_images/DJI_0421.jpg