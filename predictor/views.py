from django.shortcuts import render

# Create your views here.
import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .predictor import predict_breed
from django.core.files.storage import default_storage

logger = logging.getLogger(__name__)

class PredictDogBreed(APIView):
  def post(self, request):
    file = request.FILES.get('image')
    if not file:
      return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
    file_path = default_storage.save(file.name, file)
    try:
      breed = predict_breed(file_path)
      return Response({'breed': breed}, status=status.HTTP_200_OK)
    except Exception as e:
      logger.error(f"Prediction failed: {str(e)}")
      return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
      if default_storage.exists(file_path):
        default_storage.delete(file_path)

