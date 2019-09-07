FROM continuumio/miniconda:latest

WORKDIR /home/elekin

COPY environment.yml ./
COPY api.py ./
COPY boot.sh ./

RUN chmod +x boot.sh
RUN conda create  \
    && conda update
    && sync && conda clean --all && sync \
    && bash -c "source activate elekin \
    && rm -rf ~/.cache/pip/*

RUN bash -c 'source activate elekin'

RUN mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \'0.0.0.0\' > ~/.jupyter/jupyter_notebook_config.py

EXPOSE 8888 8888

#CMD ["/home/elekin/pyrestfmri/preprocess.py","-c","/home/elekin/pyrestfmri/conf/config_test.json"]

CMD jupyter-notebook


RUN conda env create -f environment.yml

RUN echo "source activate your-environment-name" > ~/.bashrc
ENV PATH /opt/conda/envs/your-environment-name/bin:$PATH

EXPOSE 5000

ENTRYPOINT ["./boot.sh"]