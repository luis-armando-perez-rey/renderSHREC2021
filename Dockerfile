# set base image (host OS)
FROM python:3.8
# Replicate the host user UID and GID to the image
USER root
# Identify the maintainer of an image
LABEL maintainer="l.a.perez.rey@tue.nl"
# Set the working directory in the container
WORKDIR /renderSHREC2021
# Copy the content of the local directory to the working directory
COPY ./modules ./modules
COPY ./experiments ./experiments
COPY ./data_scripts ./data_scripts
COPY run_culture .
COPY run_shape .
COPY run .
COPY requirements.txt .
# Install dependencies
RUN pip3 install -r requirements.txt
# Make run script executable
#RUN ["chmod", "+x", "./run_culture"]
#RUN ["chmod", "+x", "./run"]
#RUN ["chmod", "+x", "./run_shape"]
CMD chmod +x run_culture
CMD chmod +x run_shape
CMD chmod +x run
# In case you are not using a script to execute your project (run.sh is not present),you must comment the previous file.