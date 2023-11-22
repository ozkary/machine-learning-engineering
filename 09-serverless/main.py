# create a function 'predict' and use the class BeesWaspsModel from the bees-wasps.py to download and preprocess the image
from img_ai.__version__ import __version__
from img_ai.bees_wasps import BeesWaspsModel

def predict(url):

    # from local folder
    # bees_wasps_model = BeesWaspsModel('./models/bees-wasps.tflite')
    
    # from the docker image
    bees_wasps_model = BeesWaspsModel('bees-wasps-v2.tflite')
    
    # download the image
    img_normalized = bees_wasps_model.download_and_preprocess_image(url, (150,150))
    
    # prepare the image
    img = bees_wasps_model.img_inference(img_normalized)
        
    return img

def main(event, context):
    url = event['url']    
    result = predict(url)
    return result

