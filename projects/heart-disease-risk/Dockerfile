# Use the base image
FROM svizor/zoomcamp-model:3.10.12-slim

# Set the working directory
WORKDIR /app

# Copy the Pipenv files to the container
COPY Pipfile Pipfile.lock /app/

# Install pipenv and dependencies
RUN pip install pipenv
RUN pipenv install --system --deploy

# copy the bin files
COPY bin/ /app/bin/

# Copy the Flask script to the container
COPY data_predict.py /app/
COPY app.py /app/

# Expose the port your Flask app runs on
EXPOSE 8000

# Run the Flask app with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "4"]
