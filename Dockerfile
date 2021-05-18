FROM pytorch/pytorch:latest

# Setup shared volume
VOLUME /var

# Create the working directory.
RUN mkdir /workdir
WORKDIR /workdir

# Install libraries
RUN add-apt-repository ppa:mc3man/trusty-media
RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Setup pipenv
ADD requirements.txt /workdir
RUN pip install -r requirements.txt

ADD . /workdir
CMD python train.py --log_dir /var/runs/exp1
