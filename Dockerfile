# Use official minimal Python image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project code into the container
COPY . .

# Expose port FastAPI will run on 
EXPOSE 8000

# Command to run FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
