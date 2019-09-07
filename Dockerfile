FROM continuumio/miniconda:latest

WORKDIR /home/elekin

RUN chmod +x boot.sh
RUN conda env create  -f resources/envs/environment.yml && conda update --all
RUN conda activate elekin

RUN mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \'0.0.0.0\' > ~/.jupyter/jupyter_notebook_config.py
#CMD ["/home/elekin/pyrestfmri/preprocess.py","-c","/home/elekin/pyrestfmri/conf/config_test.json"]
CMD jupyter-notebook

EXPOSE 5000
EXPOSE 8888 8888

ENTRYPOINT ["./boot.sh"]