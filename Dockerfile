FROM conda/miniconda3:latest

COPY commtf.yml .
RUN conda env create -f commtf.yml

SHELL ["conda", "run", "-n", "commtf", "/bin/bash", "-c"]

WORKDIR /workspace