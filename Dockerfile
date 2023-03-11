FROM ubuntu:20.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y vim wget python3 python3-pip &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install numpy \
    torch \
    timm \
    torchsummary \
    pandas \
    scipy \
    sklearn \
    scikit-learn

COPY codes ./
COPY data ../data
COPY images ../images
