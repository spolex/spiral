FROM continuumio/miniconda:latest

COPY resources/envs/environment.yml environment.yml

RUN apt-get update && apt-get upgrade -y\
 && apt-get install -y -q --no-install-recommends \
           gcc \
           g++ \
           tree \
           vim \
           nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda env create  -f environment.yml
RUN echo "conda activate $(head -1 environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 environment.yml | cut -d' ' -f2)/bin:$PATH

EXPOSE 5000
EXPOSE 8888 8888
RUN mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \'0.0.0.0\' > ~/.jupyter/jupyter_notebook_config.py
