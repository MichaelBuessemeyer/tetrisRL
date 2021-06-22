#FROM tensorflow/tensorflow:latest-gpu  
FROM nvidia/cuda
RUN  apt update
RUN apt install -y python3.8 g++ make build-essential python3-pip apt-transport-https curl gnupg
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
RUN mv bazel.gpg /etc/apt/trusted.gpg.d/
RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN apt update
RUN apt install -y bazel
# RUN apt -y full-upgrade
RUN python3 -m pip install --upgrade pip
WORKDIR /tetrisRL 
RUN mkdir logs
COPY ./requirements.txt .
RUN python3 -m pip install -r requirements.txt
