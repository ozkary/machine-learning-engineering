# Use the official Azure Functions Python image
FROM mcr.microsoft.com/azure-functions/python:3.0-python3.8

# Set the working directory to your function app
WORKDIR /home/site/wwwroot

# Copy the Azure Functions code and configuration to the container
COPY . .

# Install dependencies for each function

# for multiple functions add the folder names separated by space
# RUN for func_folder in predict train validate; do \

RUN for func_folder in predict; do \
        cd $func_folder && \
        pip install --upgrade pip && \
        pip install -r requirements.txt && \
        cd ..; \
    done

# Expose port 80 for the Azure Functions runtime
EXPOSE 80

# Start the Azure Functions runtime
CMD ["azure-functions", "start"]
