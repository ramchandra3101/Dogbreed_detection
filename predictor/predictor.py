import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import logging
from django.core.files.storage import default_storage
import os

# Configure logging
logger = logging.getLogger(__name__)

# Path to your trained model
MODEL_PATH = "/app/dog_breed_classifier_model.h5"

# Load your trained model with custom objects if necessary
def load_model_with_custom_objects():
    custom_objects = {
            0: 'n02085620-Chihuahua',
            1: 'n02085782-Japanese_spaniel',
            2: 'n02085936-Maltese_dog',
            3: 'n02086079-Pekinese',
            4: 'n02086240-Shih-Tzu',
            5: 'n02086646-Blenheim_spaniel',
            6: 'n02086910-papillon',
            7: 'n02087046-toy_terrier',
            8: 'n02087394-Rhodesian_ridgeback',
            9: 'n02088094-Afghan_hound',
            10: 'n02088238-basset',
            11: 'n02088364-beagle',
            12: 'n02088466-bloodhound',
            13: 'n02088632-bluetick',
            14: 'n02089078-black-and-tan_coonhound',
            15: 'n02089867-Walker_hound',
            16: 'n02089973-English_foxhound',
            17: 'n02090379-redbone',
            18: 'n02090622-borzoi',
            19: 'n02090721-Irish_wolfhound',
            20: 'n02091032-Italian_greyhound',
            21: 'n02091134-whippet',
            22: 'n02091244-Ibizan_hound',
            23: 'n02091467-Norwegian_elkhound',
            24: 'n02091635-otterhound',
            25: 'n02091831-Saluki',
            26: 'n02092002-Scottish_deerhound',
            27: 'n02092339-Weimaraner',
            28: 'n02093256-Staffordshire_bullterrier',
            29: 'n02093428-American_Staffordshire_terrier',
            30: 'n02093647-Bedlington_terrier',
            31: 'n02093754-Border_terrier',
            32: 'n02093859-Kerry_blue_terrier',
            33: 'n02093991-Irish_terrier',
            34: 'n02094114-Norfolk_terrier',
            35: 'n02094258-Norwich_terrier',
            36: 'n02094433-Yorkshire_terrier',
            37: 'n02095314-wire-haired_fox_terrier',
            38: 'n02095570-Lakeland_terrier',
            39: 'n02095889-Sealyham_terrier',
            40: 'n02096051-Airedale',
            41: 'n02096177-cairn',
            42: 'n02096294-Australian_terrier',
            43: 'n02096437-Dandie_Dinmont',
            44: 'n02096585-Boston_bull',
            45: 'n02097047-miniature_schnauzer',
            46: 'n02097130-giant_schnauzer',
            47: 'n02097209-standard_schnauzer',
            48: 'n02097298-Scotch_terrier',
            49: 'n02097474-Tibetan_terrier',
            50: 'n02097658-silky_terrier',
            51: 'n02098105-soft-coated_wheaten_terrier',
            52: 'n02098286-West_Highland_white_terrier',
            53: 'n02098413-Lhasa',
            54: 'n02099267-flat-coated_retriever',
            55: 'n02099429-curly-coated_retriever',
            56: 'n02099601-golden_retriever',
            57: 'n02099712-Labrador_retriever',
            58: 'n02099849-Chesapeake_Bay_retriever',
            59: 'n02100236-German_short-haired_pointer',
            60: 'n02100583-vizsla',
            61: 'n02100735-English_setter',
            62: 'n02100877-Irish_setter',
            63: 'n02101006-Gordon_setter',
            64: 'n02101388-Brittany_spaniel',
            65: 'n02101556-clumber',
            66: 'n02102040-English_springer',
            67: 'n02102177-Welsh_springer_spaniel',
            68: 'n02102318-cocker_spaniel',
            69: 'n02102480-Sussex_spaniel',
            70: 'n02102973-Irish_water_spaniel',
            71: 'n02104029-kuvasz',
            72: 'n02104365-schipperke',
            73: 'n02105056-groenendael',
            74: 'n02105162-malinois',
            75: 'n02105251-briard',
            76: 'n02105412-kelpie',
            77: 'n02105505-komondor',
            78: 'n02105641-Old_English_sheepdog',
            79: 'n02105855-Shetland_sheepdog',
            80: 'n02106030-collie',
            81: 'n02106166-Border_collie',
            82: 'n02106382-Bouvier_des_Flandres',
            83: 'n02106550-Rottweiler',
            84: 'n02106662-German_shepherd',
            85: 'n02107174-Doberman',
            86: 'n02107312-miniature_pinscher',
            87: 'n02107574-Greater_Swiss_Mountain_dog',
            88: 'n02107683-Bernese_mountain_dog',
            89: 'n02107908-Appenzeller',
            90: 'n02108000-EntleBucher',
            91: 'n02108089-boxer',
            92: 'n02108422-bull_mastiff',
            93: 'n02108551-Tibetan_mastiff',
            94: 'n02108915-French_bulldog',
            95: 'n02109047-Great_Dane',
            96: 'n02109525-Saint_Bernard',
            97: 'n02109961-Eskimo_dog',
            98: 'n02110063-malamute',
            99: 'n02110185-Siberian_husky',
            100: 'n02110627-affenpinscher',
            101: 'n02110806-basenji',
            102: 'n02110958-pug',
            103: 'n02111129-Leonberg',
            104: 'n02111277-Newfoundland',
            105: 'n02111500-Great_Pyrenees',
            106: 'n02111889-Samoyed',
            107: 'n02112018-Pomeranian',
            108: 'n02112137-chow',
            109: 'n02112350-keeshond',
            110: 'n02112706-Brabancon_griffon',
            111: 'n02113023-Pembroke',
            112: 'n02113186-Cardigan',
            113: 'n02113624-toy_poodle',
            114: 'n02113712-miniature_poodle',
            115: 'n02113799-standard_poodle',
            116: 'n02113978-Mexican_hairless',
            117: 'n02115641-dingo',
            118: 'n02115913-dhole',
            119: 'n02116738-African_hunting_dog'
    }
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Load the model once globally
model = load_model_with_custom_objects()

# Image size that the model expects
IMG_SIZE = (224, 224)

# Class indices (map from predicted label index to actual breed name)
class_indices = {
    0: 'Chihuahua',
    1: 'Japanese Spaniel',
    2: 'Maltese Dog',
    3: 'Pekinese',
    4: 'Shih Tzu',
    5: 'Blenheim Spaniel',
    6: 'Papillon',
    7: 'Toy Terrier',
    8: 'Rhodesian Ridgeback',
    9: 'Afghan Hound',
    10: 'Basset',
    11: 'Beagle',
    12: 'Bloodhound',
    13: 'Bluetick',
    14: 'Black and Tan Coonhound',
    15: 'Walker Hound',
    16: 'English Foxhound',
    17: 'Redbone',
    18: 'Borzoi',
    19: 'Irish Wolfhound',
    20: 'Italian Greyhound',
    21: 'Whippet',
    22: 'Ibizan Hound',
    23: 'Norwegian Elkhound',
    24: 'Otterhound',
    25: 'Saluki',
    26: 'Scottish Deerhound',
    27: 'Weimaraner',
    28: 'Staffordshire Bullterrier',
    29: 'American Staffordshire Terrier',
    30: 'Bedlington Terrier',
    31: 'Border Terrier',
    32: 'Kerry Blue Terrier',
    33: 'Irish Terrier',
    34: 'Norfolk Terrier',
    35: 'Norwich Terrier',
    36: 'Yorkshire Terrier',
    37: 'Wire-Haired Fox Terrier',
    38: 'Lakeland Terrier',
    39: 'Sealyham Terrier',
    40: 'Airedale',
    41: 'Cairn',
    42: 'Australian Terrier',
    43: 'Dandie Dinmont',
    44: 'Boston Bull',
    45: 'Miniature Schnauzer',
    46: 'Giant Schnauzer',
    47: 'Standard Schnauzer',
    48: 'Scotch Terrier',
    49: 'Tibetan Terrier',
    50: 'Silky Terrier',
    51: 'Soft-Coated Wheaten Terrier',
    52: 'West Highland White Terrier',
    53: 'Lhasa',
    54: 'Flat-Coated Retriever',
    55: 'Curly-Coated Retriever',
    56: 'Golden Retriever',
    57: 'Labrador Retriever',
    58: 'Chesapeake Bay Retriever',
    59: 'German Short-Haired Pointer',
    60: 'Vizsla',
    61: 'English Setter',
    62: 'Irish Setter',
    63: 'Gordon Setter',
    64: 'Brittany Spaniel',
    65: 'Clumber',
    66: 'English Springer',
    67: 'Welsh Springer Spaniel',
    68: 'Cocker Spaniel',
    69: 'Sussex Spaniel',
    70: 'Irish Water Spaniel',
    71: 'Kuvasz',
    72: 'Schipperke',
    73: 'Groenendael',
    74: 'Malinois',
    75: 'Briard',
    76: 'Kelpie',
    77: 'Komondor',
    78: 'Old English Sheepdog',
    79: 'Shetland Sheepdog',
    80: 'Collie',
    81: 'Border Collie',
    82: 'Bouvier des Flandres',
    83: 'Rottweiler',
    84: 'German Shepherd',
    85: 'Doberman',
    86: 'Miniature Pinscher',
    87: 'Greater Swiss Mountain Dog',
    88: 'Bernese Mountain Dog',
    89: 'Appenzeller',
    90: 'EntleBucher',
    91: 'Boxer',
    92: 'Bull Mastiff',
    93: 'Tibetan Mastiff',
    94: 'French Bulldog',
    95: 'Great Dane',
    96: 'Saint Bernard',
    97: 'Eskimo Dog',
    98: 'Malamute',
    99: 'Siberian Husky',
    100: 'Affenpinscher',
    101: 'Basenji',
    102: 'Pug',
    103: 'Leonberg',
    104: 'Newfoundland',
    105: 'Great Pyrenees',
    106: 'Samoyed',
    107: 'Pomeranian',
    108: 'Chow',
    109: 'Keeshond',
    110: 'Brabancon Griffon',
    111: 'Pembroke',
    112: 'Cardigan',
    113: 'Toy Poodle',
    114: 'Miniature Poodle',
    115: 'Standard Poodle',
    116: 'Mexican Hairless',
    117: 'Dingo',
    118: 'Dhole',
    119: 'African Hunting Dog'   
    
}

# Labels for mapping predictions back to breed names
labels = dict((v, k) for k, v in class_indices.items())

# Function to preprocess the image before prediction
def preprocess_image(image_path):
    try:
        img = tf_image.load_img(image_path, target_size=IMG_SIZE)
        img_array = tf_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

# Prediction function to predict breed from an image
def predict_breed(image_path):
    try:
        # Preprocess the image
        img_array = preprocess_image(image_path)
        print(image_path)
        
        # Perform prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_breed = class_indices[predicted_class_index]
        
        # Log predictions and index
        logger.info(f"Predicted class index: {predicted_class_index}")
        logger.info(f"Predicted breed: {predicted_breed}")

        # Top 3 predictions for more insights
        top_3_preds = np.argsort(predictions[0])[-3:][::-1]
        top_3_breeds = [(class_indices[i], float(predictions[0][i] * 100)) for i in top_3_preds]
        print(predicted_breed,top_3_breeds)

        # Return predictions as a dictionary
        return {
            "predicted_breed": predicted_breed,
            "top_3_breeds": top_3_breeds
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise
