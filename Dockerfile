FROM nvidia/cuda:10.1-base-ubuntu18.04

RUN apt-get update; \
    apt-get install -y software-properties-common; \
    add-apt-repository ppa:deadsnakes/ppa; \
    apt-get install python3.7 curl -y ; \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py ; \
    python3 get-pip.py ; \
    pip3 install torch torchvision numpy

RUN apt-get install python3-tk -y

RUN pip3 install awscli

COPY . /usr/local/rl_hockey

WORKDIR /usr/local/rl_hockey
