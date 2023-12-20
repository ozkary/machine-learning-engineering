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
COPY models/ /app/models/

# copy the data files
COPY data/interaction_types.csv /app/data/
COPY data/drugbank_pca50.csv.gz /app/data/

# Copy the DDI library
COPY ddi_lib/ /app/ddi_lib/

# Copy the Flask script to the container
COPY app.py /app/

# Expose the port the Flask app runs on
EXPOSE 8000

# Run the Flask app with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "4"]