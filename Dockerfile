FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

SHELL ["/bin/bash", "-c"]
RUN apt update -y
RUN apt install git build-essential cmake g++ libboost-dev libboost-system-dev libboost-filesystem-dev -y
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
RUN pip install torch-summary
RUN pip install Cython
RUN pip install numpy scikit-learn networkx scipy
RUN pip install persim
RUN pip install pandas
RUN pip install ipywidgets
RUN pip install torchinfo
RUN pip install tensorboard
RUN pip install ripser
RUN pip install ogb
RUN pip install torchmetrics