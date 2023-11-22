# Use the base image
FROM agrigorev/zoomcamp-bees-wasps:v2

#define a working directory
WORKDIR .

# Set the working directory azure functions
# WORKDIR /home/site/wwwroot

# ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    # AzureFunctionsJobHost__Logging__Console__IsEnabled=true

# Copy the script to the container
# COPY *.py /home/site/wwwroot
COPY main.py .
COPY img_ai ./img_ai/

# Copy the Pipenv files to the container
# COPY Pipfile Pipfile.lock ./

# Install pipenv and dependencies
# RUN pip install pipenv
# RUN pipenv install --system --deploy
RUN pip install numpy
RUN pip install Pillow
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl

# use as the entry point that should be called by the serverless functions
CMD ["main.main"]

# build the image
# docker build -t ozkary/bees-wasps:v1 .
# run from the container
# docker run -p 8080:8080 -it ozkary/bees-wasps:v1
# check files in container
# docker exec -it ozkary/bees-wasps:v1 ls .
