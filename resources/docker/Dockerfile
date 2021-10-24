# Start from a core stack version
FROM tensorflow/tensorflow:2.5.0-gpu-jupyter

USER root

RUN apt update && apt install --yes git

USER $NB_USER

ADD requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /home/jovyan/spiral

