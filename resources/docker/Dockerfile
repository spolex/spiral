# Start from a core stack version
FROM jupyter/base-notebook

USER root

RUN apt update && apt install --yes git

USER $NB_USER

ADD ./envs/requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /home/jovyan/spiral

