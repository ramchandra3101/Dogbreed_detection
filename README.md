docker volume create model_volume

docker run --rm -v model_volume:/app -v "copy yor path to model file":/model busybox cp /model/dog_breed_classifier_model.h5 /app/dog_breed_classifier_model.h5

docker-compose up --build
docker-compose up
