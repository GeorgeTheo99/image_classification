#!/bin/bash


bash ./predict --docker-image nima-cpu \
--base-model-name MobileNet \
--weights-file $(pwd)/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5 \
--image-source $(pwd)/src/tests/test_images/DSC_9692.jpg

