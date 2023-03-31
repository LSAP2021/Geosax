FROM tensorflow/tensorflow:latest-gpu-py3
LABEL maintainer "Armel Nya< cudatools@nvidia.com> "
RUN  apt-get update \ 
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*
RUN apt-get install -y python3
RUN pip install --upgrade pip
RUN apt-get update && yes|apt-get upgrade 
RUN apt-get install -y wget bzip2
RUN pip install pandas 
RUN pip  install vtk 
RUN pip install Pillow  
RUN pip install ctgan
RUN pip install graphviz 
RUN pip install sdv 
RUN pip install scikit-image 
RUN pip install pydot 
RUN pip install opencv-python 
RUN pip install numpy 
RUN pip install tqdm
RUN pip install ttcrpy
COPY requirements.txt /tmp
ENV PYTHONPATH "${PYTHONPATH}:/PAIRGEMM/"
WORKDIR /home/armel/PAIREMM
COPY . /home/armel/PAIREMM/Dockerfile/main.py
ENTRYPOINT ["python3"]
