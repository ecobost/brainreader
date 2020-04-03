# Run as:
#   docker build --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" -t ecobost/brainreader .

############ Intermediate container to copy private repos
FROM nvidia/cuda:10.1-cudnn7-devel as intermediate

# Install needed libraries
RUN apt update && \
    apt install -y openssh-client git

# Authorize SSH Host
RUN mkdir /root/.ssh && \
    ssh-keyscan github.com > /root/.ssh/known_hosts

# Add SSH private key to the ssh agent inside this image
ARG SSH_PRIVATE_KEY
RUN echo "$SSH_PRIVATE_KEY" > /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa

RUN git clone git@github.com:cajal/featurevis.git /src/featurevis


########### Final Repo
FROM nvidia/cuda:10.1-cudnn7-devel

ARG DEBIAN_FRONTEND=noninteractive
LABEL MANTAINER="Erick Cobos <ecobos@bcm.edu>"
WORKDIR /src

# Copy private repos
COPY --from=intermediate /src/featurevis /src/featurevis

# Upgrade system 
RUN apt update && apt upgrade -y

# Install Python 3
RUN apt update && \
    apt install -y python3-dev python3-pip python3-tk && \
    pip3 install numpy scipy matplotlib jupyterlab

# Install pytorch 
RUN pip3 install torch torchvision

# Install datajoint
RUN pip3 install datajoint

# Install featurevis
RUN pip3 install -e /src/featurevis

# Install brainreader
ADD ./setup.py /src/brainreader/setup.py
ADD ./brainreader /src/brainreader/brainreader
RUN pip3 install -e /src/brainreader

# Install extra libraries (non-essential but useful)
RUN apt install -y git nano

# Clean apt lists
RUN rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/bin/bash"]
