# FROM python:3.11-slim

# RUN apt-get update && apt-get install -y git

# WORKDIR /app

# COPY my_requirements.txt .
# RUN pip install -r my_requirements.txt

# COPY . .

# CMD ["streamlit", "run", "front_end.py", "--server.enableCORS", "false", "--browser.serverAddress", "0.0.0.0", "--browser.gatherUsageStats", "false", "--server.port", "8080"]
FROM python:3.11-slim

# Install Git and Docker CLI
RUN apt-get update && apt-get install -y git docker.io

WORKDIR /app

# Copy and install Python dependencies
COPY my_requirements.txt .
RUN pip install --no-cache-dir -r my_requirements.txt

# Copy the rest of the application
COPY . .

# Streamlit specific commands for running in a container
CMD ["streamlit", "run", "front_end.py", "--server.enableCORS", "false", "--browser.serverAddress", "0.0.0.0", "--browser.gatherUsageStats", "false", "--server.port", "8080"]

# https://medium.com/ml-hobbyist/deploying-a-streamlit-app-on-google-cloud-platform-app-engine-vs-cloud-run-1625232d0363

# LOCAL TESTING
# build image
# docker build -t gcr.io/concise-result-418721/streamlit-app .

# test locally - DOESNT WORK
# docker run -p 8501:8501 gcr.io/concise-result-418721/streamlit-app
# docker run -p 8501:8501 -v /var/run/docker.sock:/var/run/docker.sock gcr.io/concise-result-418721/streamlit-app 

# PUBLISH TO SERVICE #NOT IN USE
# build the container
# gcloud builds submit --tag gcr.io/concise-result-418721/streamlit-app

# deploy the container #NOT IN USE
# gcloud run deploy --image gcr.io/concise-result-418721/streamlit-app --platform managed --allow-unauthenticated