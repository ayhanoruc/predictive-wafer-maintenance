#base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc


# Copy the current directory contents into the container at /usr/src/app
COPY . .


# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make sure scripts in .local are usable (if you're installing with --user)
ENV PATH=/root/.local/bin:$PATH

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run Gunicorn to serve the app
CMD ["gunicorn", "--workers=3", "--bind", "0.0.0.0:5000", "app:app"]
# 3 worker processes to handle requests.
#bind the workers to all network interfaces on port 5000 so that the container can receive web requests.
#"app:app": This tells Gunicorn that the WSGI application is the app object inside the app.py file.

#CMD ["python3", "app.py"]