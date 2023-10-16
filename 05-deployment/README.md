# Machine Learning (ML) Deployment

Deploying a machine learning model involves making your trained model available for use in a production environment where it can generate predictions or assist in decision-making. Here's an overview of the deployment process:

1. **Model Serialization**: Serialize your trained model into a format that can be easily stored and loaded. Common formats include pickle, joblib, or ONNX.

2. **Creating an API**: Expose your model through an API (Application Programming Interface). This API should define endpoints where clients can send requests for predictions.

3. **Model Hosting**: Host your API and the serialized model on a server or a cloud platform. Common options include cloud providers like AWS, Google Cloud, Microsoft Azure, or specialized platforms like Heroku.

4. **Integration**: Integrate the API into your application, website, or system. Clients can now send data to the API and receive predictions.

5. **Monitoring and Maintenance**: Continuously monitor your model's performance, ensure that it's providing accurate predictions, and retrain or update the model as needed to maintain its effectiveness.

6. **Scalability**: Ensure that your deployment can handle a growing number of requests. You might need to scale up resources, use load balancers, or optimize your code.

7. **Security**: Implement security measures to protect your model and data. This includes access controls, encryption, and other relevant security best practices.

8. **User Interface (Optional)**: If needed, develop a user interface to allow users to interact with the model easily.

9. **Feedback Loop**: Incorporate feedback from users and the system to improve the model over time. Retraining and updating the model based on new data can enhance its performance.

The specific approach and technologies you use for deployment can vary based on your project requirements, budget, and the scale at which you're operating. Always consider factors like scalability, reliability, and security when deploying machine learning models in a real-world setting.

## Using Pipenv

Pipenv is a package manager for Python that provides an improved way to manage project dependencies and virtual environments. It's similar to npm in the JavaScript world but for Python.

Here's how it works, drawing a parallel with npm:

1. **Dependency Management**:
   - Just like npm's `package.json`, Pipenv uses a `Pipfile` to list project dependencies and their versions.

2. **Virtual Environments**:
   - Pipenv creates and manages a virtual environment for your projects, isolating your dependencies from other projects. This is akin to `node_modules` in npm, but in a more controlled and organized manner.

3. **Installation of Dependencies**:
   - When you install a package using `pipenv install`, it's added to the `Pipfile`, similar to running `npm install --save` which updates `package.json`.
   
4. **Lock File (`Pipfile.lock`)**:
   - The lock file (`Pipfile.lock`) contains the exact versions of all dependencies, including their sub-dependencies. This ensures that the same versions are installed across different machines or environments.
   - It's similar to `package-lock.json` in npm, which pins down the exact versions of packages and their dependencies.

5. **Reproducibility**:
   - The lock file is crucial for reproducibility. When someone else works on your project, they can run `pipenv install` to install exactly the same versions of dependencies you used.

6. **Environment Activation**:
   - You activate the project's virtual environment using `pipenv shell`, similar to activating a virtual environment in Python using `source venv/bin/activate`.

In summary, Pipenv streamlines dependency management and ensures a consistent development environment by creating a virtual environment and using the `Pipfile.lock` to pin down dependency versions. It's a helpful tool to organize and manage Python projects, similar to how npm and `package.json` are used in the JavaScript ecosystem.

## Use saved models (bin)

To use the saved models (dv.bin for DictVectorizer and model1.bin for Logistic Regression) to make predictions on new data in Python, follow these steps:

1. **Load the Models and Data**:
   ```python
   import pickle

   # Load the DictVectorizer
   with open('dv.bin', 'rb') as f:
       dv = pickle.load(f)

   # Load the Logistic Regression model
   with open('model1.bin', 'rb') as f:
       model = pickle.load(f)
   
   # New data for prediction
   new_data = {"job": "retired", "duration": 445, "poutcome": "success"}
   ```

2. **Transform the New Data**:
   ```python
   # Transform the new data using the DictVectorizer
   transformed_data = dv.transform([new_data])
   ```

3. **Make Predictions**:
   ```python
   # Predict using the loaded model
   prediction = model.predict(transformed_data)
   
   # Print the prediction
   print("Predicted class:", prediction[0])
   ```

This will use the trained DictVectorizer to transform the new data and then make predictions using the loaded Logistic Regression model.

Make sure to replace `'dv.bin'` and `'model1.bin'` with the actual file paths where your models are saved. Also, ensure that the structure of the new data (`new_data`) matches the structure expected by the models (same features and format).

In binary classification with Scikit-Learn, typically the positive class (often denoted as "yes" or "1") is represented by the second index in the probability scores. Therefore, if you're interested in the probability associated with the positive class, you would look at the second value in the probability array. The first value represents the probability of the negative class.

However, it's important to note that the order can vary depending on the specific implementation or library, so it's always a good practice to confirm the order by checking the class labels and corresponding probability values in your specific use case.

```python

# Predict using the loaded model
probabilities = model.predict_proba(transformed_data)

# break down the probabilities into yes or no score
no_score, yes_score = probabilities[0]

# get the class labels from the model
no_label, yes_label = model.classes_

for i, probs in enumerate(probabilities):
    print(f"Prediction {i+1}:")
    for label, prob in zip(model.classes_, probs):
        print(f"Class {label}: Probability {prob}")

# Print the probability scores for each class
print(f'Probability the client would get a credit: {yes_label} - {yes_score}   {no_label} - {no_score}')

```

## Flask Deployment

Deploying a machine learning model using Flask and Gunicorn on a Linux server is a common approach. Here's a high-level overview of the steps you'll need to follow:

1. **Set Up Your Model:**
   Make sure your trained model and any other necessary files are ready for deployment. These might include your trained model file, the dictionary vectorizer, and any pre-processing functions.

2. **Create a Flask Web Application:**
   Create a Flask application that will serve as your web service. Define routes to handle the incoming requests and use your model to generate predictions.

3. **Handle Model Loading:**
   Load your pre-trained model and any other required components (like the dictionary vectorizer) when the Flask application starts.

4. **Define API Endpoints:**
   Define the API endpoints for your application, specifying how to handle different types of requests. For instance, you might have an endpoint to receive new data and return predictions.

5. **Deploy on a Linux Server:**
   Set up a Linux server, install Gunicorn, and configure it to run your Flask application. Gunicorn acts as a WSGI HTTP server for running your Flask app.

Here's a simple example to illustrate these steps:

### Install Flask and Gunicorn

It's a good practice to install Flask and Gunicorn within your `pipenv` virtual environment to keep your project dependencies isolated. Here's how you can do it:

1. **Activate Your Pipenv Environment:**
   Navigate to your project directory and activate your `pipenv` environment if it's not already activated.

    ```bash
    cd your_project_directory
    pipenv shell
    ```

2. **Install Flask and Gunicorn:**
   Use `pipenv` to install Flask and Gunicorn.

    ```bash
    pipenv install flask gunicorn
    ```

   This command will install Flask and Gunicorn within your `pipenv` environment.

Now, you'll have Flask and Gunicorn installed within your project's virtual environment, making them isolated from your system Python and other projects. You can proceed to develop your Flask application and use Gunicorn to serve it on a Linux server.

### Write the App Code

```python
# app.py - Your Flask application

from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the pre-trained model and other necessary components
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Process the data, make predictions using the model, and return the results
    prediction = model.predict(data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Ensure you have `gunicorn` installed on your Linux server. You can then start the application using:

```bash
gunicorn -b 0.0.0.0:8000 app:app
```

This command starts Gunicorn and binds it to address 0.0.0.0 (meaning all available network interfaces) on port 8000, running your Flask app.

Remember to secure your API appropriately if it will be publicly accessible.

This is a basic example, and real-world deployments may involve more complexities, such as handling errors, logging, authentication, etc.

### Building a Docker image with the models

This is how the Dockerfile for this image looks like:

```bash
FROM python:3.10.12-slim
WORKDIR /app
COPY ["model2.bin", "dv.bin", "./"]

```
We already built it and then pushed it to svizor/zoomcamp-model:3.10.12-slim.

To download an existen image, run this command from the terminal

> Make sure Docker Desktop is running

```bash
svizor/zoomcamp-model:3.10.12-slim
```

#### Create a new Docker file

You'll need to create a `Dockerfile` and set it up accordingly to achieve this. Here's a basic outline of how you can do it:

1. Create a `Dockerfile` with the following content:

```Dockerfile
# Use the base image
FROM svizor/zoomcamp-model:3.10.12-slim

# Set the working directory
WORKDIR /app

# Copy the Pipenv files to the container
COPY Pipfile Pipfile.lock /app/

# Install pipenv and dependencies
RUN pip install pipenv
RUN pipenv install --system --deploy

# Copy the Flask script to the container
COPY app.py /app/

# Expose the port your Flask app runs on
EXPOSE 5000

# Run the Flask app with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "4"]
```

2. In the same directory as your `Dockerfile`, make sure you have the `Pipfile`, `Pipfile.lock`, and your Flask script (`app.py`).

3. Build the Docker image using the following command (replace `your_image_tag` with a desired tag):

```bash
docker build -t your_image_tag .
```

4. Once the image is built, you can run the Docker container using:

```bash
docker run -p 5000:5000 your_image_tag
```

This will start the Flask app inside the Docker container, and it will be accessible at `http://localhost:5000`.

Make sure to adjust the necessary details according to your specific Flask app and dependencies.

### Docker Commands

To check what containers are running, you can use the `docker ps` command. This command lists all running containers along with essential information. Here's how you can use it:

```bash
docker ps
```

If you want to see all containers, including those that have exited, you can add the `-a` flag:

```bash
docker ps -a
```

The output will display details such as the container ID, image used to create the container, names, status, ports, etc. If a container is currently running, it will show its status as "Up". If a container is not running but exists, it will show its status as "Exited".

If you only want to see the container IDs, you can use the `docker ps -q` command:

```bash
docker ps -q
```

This will print only the container IDs of running containers.

Remember, container IDs are unique, so they can be used to reference specific containers for actions like stopping or removing them.

### Update a container image

When you make changes to your code, especially if they are in the application logic or dependencies, you will typically need to rebuild the Docker image to incorporate those changes.

Here's a general process to update your Docker image after making code changes:

1. **Update Your Code**: Make the necessary changes to your code.

2. **Rebuild the Docker Image**:
   - Run the Docker build command again to create a new image with the updated code:
     ```
     docker build -t your_image_name:tag .
     ```

3. **Run the Container with the Updated Image**:
   - Stop and remove the existing container based on the old image:
     ```
     docker stop container_name
     docker rm container_name
     ```
   - Run a new container using the updated image:
     ```
     docker run -d -p 8000:8000 your_image_name:tag
     ```

By following these steps, you can update your Docker image with the latest code changes and run a new container based on this updated image. Keep in mind that any changes to the Dockerfile, dependencies, or code will require a rebuild of the Docker image.