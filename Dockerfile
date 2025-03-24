# Use the specified TensorFlow GPU Jupyter image as the base image.
FROM tensorflow/tensorflow:2.11.0-gpu-jupyter

# Set the working directory.
WORKDIR /app

RUN pip install transformers
RUN pip install torch
# Copy requirements and install them.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY . .
# Expose port 5000.
EXPOSE 5100

# Run the Flask application.
CMD ["python", "app.py"]
