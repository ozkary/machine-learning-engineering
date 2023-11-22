#!/usr/bin/env python
# coding: utf-8

# # Machine Learning - Serverless Hosting
# 
# Machine Learning models can be hosted on server less functions. To host this model, we must consider the size of the packages and their security settings. 

import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np

from io import BytesIO
from urllib import request

# define a class for the bees vs wasps model inference
class BeesWaspsModel:
    def __init__(self, path):
        # load the lite model and the input/output details
        self.interpreter, self.input_details, self.output_details = self.load_lite_model(path)
    
    def load_lite_model(self, path):
        """
        Load the TensorFlow Lite model and allocate tensors.
        """
        # Load the TFLite model and allocate tensors.
        interpreter = tflite.Interpreter(model_path=path)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()    

        return interpreter, input_details, output_details
    
    def download_image(self, url):
        """
        download the image from the url
        """
        with request.urlopen(url) as resp:
            buffer = resp.read()

        stream = BytesIO(buffer)
        img = Image.open(stream)

        return img

    def prepare_image(self, img, target_size):
        """
        Resize the image to target_size i.e. (150,150)
        """
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size, Image.NEAREST)
        return img

    def preprocess_image(self, img):
        """
        convert the image to a numpy array and normalize it
        """
        # convert to numpy array
        img = np.array(img)

        # convert to float32 to avoid overflow when multiplying by 255
        img = img.astype('float32')

        # normalize to the range 0-1
        img /= 255

        return img  

    def download_and_preprocess_image(self, url, target_size):
        """
        Download the image from the url, resize it to target_size and return a numpy array
        """

        img_stream = self.download_image(url)
        img = self.prepare_image(img_stream, target_size)
        img_normalized = self.preprocess_image(img)

        return img_normalized    

    def img_inference(self, img_normalized):
        """
        Load the TensorFlow Lite model and run the inference

        def img_inference(img_normalized):
        """        
        
        # set the input tensor with the normalized image
        self.interpreter.set_tensor(self.input_details[0]['index'], [img_normalized])

        # run the inference
        self.interpreter.invoke()

        # get the output tensor
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        # # get the result value
        # result = round(output_data[0][0],3)
        
        # # print the output
        # print('Tensor Output', result)

        return output_data[0].tolist()


