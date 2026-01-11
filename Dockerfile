# Base Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY environment.yml .
# Ideally we interpret env.yml or just use pip if we had requirements.txt
# Since we have environment.yml (conda), but creating a conda image is heavy.
# Let's assume we can install packages via pip for the container to be lightweight.
# I will create a requirements.txt from the env.yml content manually here for the container primarily.
# Or better: I will use a simple pip install command for the libs we know we use.

COPY . .

# Install dependencies
RUN pip install --no-cache-dir pandas numpy scikit-learn streamlit plotly statsmodels joblib python-dotenv google-generativeai

# Expose Streamlit port
EXPOSE 8501

# Command to run
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
