# Machine Learning - Serverless Hosting

Machine Learning models can be hosted on server less functions. To host this model, we must consider the size of the packages and their security settings. 

## Serverless Hosting

Azure Functions, GCP Functions (Google) and AWS Lambda are serverless computing services that allow you to run code without provisioning or managing servers. You can use these services to host machine learning models as serverless functions, allowing you to execute code in response to events without worrying about the underlying infrastructure. Here's a brief overview of using Azure Functions and AWS Lambda for hosting machine learning models:

### Azure Functions

Azure Functions is part of the Azure serverless computing offering. You can deploy machine learning models using Azure Functions in a few steps:

1. **Choose a Runtime:**
   - Azure Functions supports multiple runtimes, including .NET, Node.js, Python, and Java. Choose the runtime that aligns with your machine learning model.

2. **Create a Function:**
   - Use the Azure portal, Visual Studio, or Azure CLI to create a new function. You can choose from various triggers (HTTP, Timer, etc.) depending on your requirements.

3. **Dependencies and Environment:**
   - Install any required dependencies for your machine learning model. For Python-based models, you might use tools like `pip` to install libraries.

4. **Code Integration:**
   - Write the code to load and run your machine learning model. You can use popular ML libraries like TensorFlow, PyTorch, or Scikit-learn.

5. **Deployment:**
   - Deploy your function to Azure using the tools provided by Azure. This can be done through the portal, Visual Studio, or the Azure CLI.

### AWS Lambda:

AWS Lambda is part of the AWS serverless computing ecosystem. Here's how you can host machine learning models using Lambda:

1. **Select a Runtime:**
   - AWS Lambda supports multiple runtimes, including Node.js, Python, Java, and more. Choose the runtime that is compatible with your machine learning model.

2. **Create a Lambda Function:**
   - Use the AWS Management Console or AWS CLI to create a new Lambda function. You can define triggers such as API Gateway, S3 events, or others.

3. **Dependencies and Environment:**
   - Specify any dependencies or libraries required for your machine learning model in your deployment package. For Python, you might use `pip` to install libraries.

4. **Code Integration:**
   - Write the code to load and execute your machine learning model. You can use popular ML libraries compatible with the chosen runtime.

5. **Deployment:**
   - Upload your deployment package to AWS Lambda using the AWS Management Console, AWS CLI, or an automation tool like AWS SAM (Serverless Application Model).

#### Considerations:

- **Cold Starts:**
  - Both Azure Functions and AWS Lambda may experience "cold starts," where the first invocation of a function has higher latency. This is something to consider, especially for real-time applications.

- **Resource Limits:**
  - Be aware of resource limits imposed by each platform, such as maximum execution time, memory, and payload size.

- **Integration with Other Services:**
  - Leverage other services within the respective cloud platforms for data storage, logging, and monitoring, as needed.

Google Cloud Functions is another serverless computing service provided by Google Cloud Platform (GCP). You can use Google Cloud Functions to host machine learning models in a serverless environment. Here's an overview of using Google Cloud Functions for this purpose:

### Google Cloud Functions:

1. **Select a Runtime:**
   - Google Cloud Functions supports multiple runtimes, including Node.js, Python, Go, and more. Choose the runtime that aligns with your machine learning model.

2. **Create a Function:**
   - Use the Google Cloud Console, `gcloud` command-line tool, or other deployment methods to create a new Cloud Function. You can define triggers such as HTTP, Cloud Storage events, Pub/Sub messages, etc.

3. **Dependencies and Environment:**
   - Specify any dependencies or libraries required for your machine learning model. For Python, you might use `pip` to install necessary packages.

4. **Code Integration:**
   - Write the code to load and execute your machine learning model. Similar to Azure Functions and AWS Lambda, you can use popular ML libraries compatible with the chosen runtime.

5. **Deployment:**
   - Deploy your function to Google Cloud Functions using the Cloud Console, `gcloud` command-line tool, or continuous integration tools.

#### Additional Considerations:

- **Cold Starts:**
  - Like Azure Functions and AWS Lambda, Google Cloud Functions may experience initial latency known as "cold starts."

- **Resource Limits:**
  - Be aware of resource limits such as maximum execution time, memory, and payload size.

- **Integration with Other GCP Services:**
  - Integrate your Cloud Function with other GCP services like Cloud Storage, BigQuery, or AI Platform for enhanced functionality.

- **Monitoring and Logging:**
  - Leverage GCP's monitoring and logging tools to keep track of function invocations, errors, and performance metrics.

- **Authentication and Authorization:**
  - Ensure that your function is properly authenticated and authorized to access any required resources or APIs.

### Choosing the Right Service:

When deciding between Azure Functions, AWS Lambda, and Google Cloud Functions for hosting machine learning models, consider factors such as familiarity with the platform, specific features offered, pricing, and integration capabilities with other cloud services. Each platform has its strengths, and the choice may also depend on your organization's existing cloud infrastructure and preferences.

## TensorFlow Lite

TensorFlow Lite is a lightweight and mobile-friendly version of the TensorFlow deep learning framework designed for resource-constrained devices, such as mobile phones, IoT devices, and edge devices. It allows you to deploy machine learning models on devices with lower computational power and memory resources compared to traditional servers or desktops. TensorFlow Lite is optimized for inference and typically used for on-device machine learning applications.

Here's a basic overview of TensorFlow Lite and how you can convert a TensorFlow model to TensorFlow Lite format:

### TensorFlow Lite:

1. **Model Optimization:**
   - TensorFlow Lite includes tools for optimizing and converting TensorFlow models to a format suitable for deployment on edge devices. This conversion process often involves quantization, which reduces the precision of the model's weights and activations to use fewer bits.

2. **Compatibility:**
   - TensorFlow Lite models are compatible with various hardware accelerators, including those commonly found in mobile devices, such as GPUs and TPUs.

3. **Interpreter:**
   - TensorFlow Lite provides a lightweight interpreter that allows you to run inference on TensorFlow Lite models on edge devices.

### Converting a TensorFlow Model to TensorFlow Lite:

Assuming you have a trained TensorFlow model in a format like SavedModel or Keras, here's a general process to convert it to TensorFlow Lite:

1. **Install TensorFlow and TensorFlow Lite Converter:**
   - Make sure you have TensorFlow installed. You may also need to install the TensorFlow Lite Converter, which provides tools for model conversion.

   ```bash
   pip install tensorflow
   pip install tflite-model-maker
   ```

2. **Convert the Model:**
   - Use the TensorFlow Lite Converter to convert your model to TensorFlow Lite format. Here's an example for a SavedModel:

   ```python
   import tensorflow as tf
   from tensorflow.keras.models import load_model
   
   # load the model file
   model = load_model("path/keras_model.h5")
   
   # Convert the model to TensorFlow Lite format   
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   tflite_model = converter.convert()

   # Save the converted model to a file
   with open("model.tflite", "wb") as f:
       f.write(tflite_model)
   ```

   Note: Adjust the paths and model loading based on your specific setup.

3. **Optimization (Optional):**
   - You can apply additional optimizations, such as quantization, to reduce the model size further. This step is optional but can be crucial for deploying models on resource-constrained devices.

4. **Deploy to Azure Functions:**
   - Once you have the TensorFlow Lite model, you can deploy it to Azure Functions. Follow the steps mentioned earlier for creating an Azure Function, and incorporate the TensorFlow Lite model loading and inference code into your function.

Remember to consider the size of the TensorFlow Lite model and the inference latency when deploying on Azure Functions, as serverless platforms have constraints on the size of deployed functions and may have variable cold start latencies.

## Use the model to run inference on a new image

![Wasp or Bee](https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg)

### Download and convert the image to TensorFlow Lite

Certainly! This code defines a series of functions for downloading, preparing, and preprocessing an image. Let's go through each function:

- `download_image(url)`

This function takes a URL as input, downloads the image from that URL, and returns a PIL (Python Imaging Library) `Image` object.

- **Parameters:**
  - `url`: The URL from which the image should be downloaded.

- **Return:**
  - Returns a PIL `Image` object representing the downloaded image.

- `prepare_image(img, target_size)`

This function takes a PIL `Image` object and resizes it to a specified target size while ensuring that the image mode is set to RGB.

- **Parameters:**
  - `img`: A PIL `Image` object representing the image.
  - `target_size`: A tuple specifying the target size (width, height) to which the image should be resized.

- **Return:**
  - Returns the prepared PIL `Image` object.

- `preprocess_image(img)`

This function converts a PIL `Image` object to a NumPy array, ensures the data type is `float32`, and normalizes the pixel values to the range [0, 1].

- **Parameters:**
  - `img`: A PIL `Image` object.

- **Return:**
  - Returns a NumPy array representing the preprocessed image.

### Usage Example:

```python
# download the image
image_url = 'https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg'

img_stream = download_image(image_url)
img = prepare_image(img_stream, target_size=(150, 150))

# convert the image to numpy array
img_normalized = preprocess_image(img)

# print the first pixel on the R channel (normalized)
print(img_normalized[0, 0, 0])
    
```

- Putting it all together

```python

from io import BytesIO
from urllib import request
from PIL import Image  

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_image(img):
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

```

### Run Image Classification

To apply the TensorFlow Lite model to a new image, you need to follow these general steps:

Certainly! This code defines a function `load_lite_model` to load a TensorFlow Lite (TFLite) model and allocate tensors. It then demonstrates how to use this function to perform inference on a normalized image using the loaded TFLite model. Let's break down the code:

### `load_lite_model(path)`

This function takes the file path of a TensorFlow Lite model as input, loads the model, allocates tensors, and returns the interpreter along with input and output details.

```python
def load_lite_model(path):
    """
    Load the TensorFlow Lite model and allocate tensors.
    """
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()    

    return interpreter, input_details, output_details
```

- **Parameters:**
  - `path`: The file path to the TensorFlow Lite model.

- **Return:**
  - Returns a tuple containing the TFLite interpreter, input details, and output details.

### Inference using the Loaded Model:

```python
# Load the TFLite model and the input/output details
interpreter, input_details, output_details = load_lite_model('./models/bees-wasps.tflite')

# Assuming you have a normalized image stored in img_normalized
# Set the input tensor with the normalized image
interpreter.set_tensor(input_details[0]['index'], [img_normalized])

# Run the inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Print the output
print('Tensor Output', round(output_data[0][0], 3))
```

- **Set Input Tensor:**
  - The code sets the input tensor of the TFLite model with the normalized image. The index of the input tensor is obtained from `input_details`.

- **Run Inference:**
  - `interpreter.invoke()` is used to run the inference.

- **Get Output Tensor:**
  - The output tensor is obtained using `interpreter.get_tensor(output_details[0]['index'])`.

- **Print Output:**
  - The final output is printed, and in this case, it's rounded to three decimal places.

This script assumes that you have a TensorFlow Lite model file located at `'./models/bees-wasps.tflite'` and a preprocessed (normalized) image stored in the `img_normalized` variable. Adjust the paths and variables based on your actual model and image data.

- Putting it all together:

```python
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the TensorFlow Lite model
def load_lite_model(path):
    """
    Load the TensorFlow Lite model and allocate tensors.
    """
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()    

    return interpreter, input_details, output_details

# load the lite model and the input/output details
interpreter, input_details, output_details = load_lite_model('./models/bees-wasps.tflite')

# set the input tensor with the normalized image
interpreter.set_tensor(input_details[0]['index'], [img_normalized])

# run the inference
interpreter.invoke()

# get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# print the output
print('Tensor Output',round(output_data[0][0],3))

```
