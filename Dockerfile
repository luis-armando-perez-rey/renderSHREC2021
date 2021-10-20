# set base image (host OS)
FROM python:3.8
# Replicate the host user UID and GID to the image
ARG UNAME=root
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME
# Identify the maintainer of an image
LABEL maintainer="l.a.perez.rey@tue.nl"
# Set the working directory in the container
WORKDIR /renderSHREC2021
# Copy the content of the local directory to the working directory
COPY ./modules .
COPY ./experiments .
COPY ./data_scripts .
COPY run_culture .
COPY run_shape .
COPY requirements.txt .
# Install dependencies
RUN pip3 install -r requirements.txt
# Make run script executable
CMD chmod +x run_culture
CMD chmod +x run_shape
# In case you are not using a script to execute your project (run.sh is not present),you must comment the previous file.