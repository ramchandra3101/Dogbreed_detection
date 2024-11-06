#!/bin/sh
if [ ! -f /app/dog_breed_classifier_model.h5 ]; then
  echo "Copying model file to Docker volume..."
  cp /path/to/local/model/dog_breed_classifier_model.h5 /app/dog_breed_classifier_model.h5
fi

