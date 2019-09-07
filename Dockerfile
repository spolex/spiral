FROM continuumio/miniconda:latest

WORKDIR /home/elekin

COPY resources/envs/environment.yml environment.yml

RUN apt-get update && apt-get upgrade -y\
# && apt-get install -f python3-dev -y\
 && apt-get install -y -q --no-install-recommends \
           gcc \
           g++ \
           tree \
           vim \
           nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda env create  -f environment.yml && conda update --all

RUN test "$(getent passwd elekin)" || useradd --no-user-group --create-home --shell /bin/bash elekin

RUN chown elekin -R /home/elekin
USER elekin

ENV CONDA_DIR="/opt/miniconda-latest" \
    PATH="/opt/miniconda-latest/bin:$PATH"
RUN bash -c 'source activate elekin'

EXPOSE 5000
EXPOSE 8888 8888
RUN mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \'0.0.0.0\' > ~/.jupyter/jupyter_notebook_config.py

CMD [ "jupyter", "lab" ]
