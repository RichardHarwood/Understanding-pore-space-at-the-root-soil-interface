# Container for running inference of
# https://github.com/RichardHarwood/Understanding-pore-space-at-the-root-soil-interface

#To build this file:
#sudo docker build . -t sydneyinformaticshub/soil

#To run this, mounting your current host directory in the container directory,
# at /project, and excute an example run:
#sudo docker run -it -v `pwd`:/project sydneyinformaticshub/soil /bin/bash -c "cd /project && python test.py"

#To push to docker hub:
#sudo docker push sydneyinformaticshub/soil

#To build a singularity container
#export SINGULARITY_CACHEDIR=`pwd`
#export SINGULARITY_TMPDIR=`pwd`
#singularity build soil.img docker://sydneyinformaticshub/soil

#To run the singularity image (noting singularity mounts the current folder by default)
#singularity run --bind /project:/project soil.img /bin/bash -c "cd "$PBS_O_WORKDIR" && python test.py"

# Pull base image.
FROM ubuntu:16.04
MAINTAINER Nathaniel Butterworth USYD SIH

# Make Artmeis specific directories that are auto-mounted
RUN mkdir /scratch /project

# Install Ubuntu packages required
RUN apt-get update -y && \
	apt-get install git curl build-essential libsm6 libxext6 libxrender-dev libgl1 libfreetype6-dev -y && \
	rm -rf /var/lib/apt/lists/*

# Install miniconda Python
WORKDIR /build

RUN curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh &&\
	mkdir /build/.conda && \
	bash miniconda.sh -b -p /build/miniconda3 &&\
	rm -rf /miniconda.sh

ENV PATH="/build/miniconda3/bin:${PATH}"
ARG PATH="/build/miniconda3/bin:${PATH}"
RUN conda install pip
RUN pip install --upgrade pip

# Install all Python packages required
# newversions: versions that seem to be the highest workable
RUN pip install numpy==1.24.4 scipy==1.10.1 pandas==2.0.3 Pillow==10.2.0 imageio==2.34.0 scikit-learn==1.3.2 scikit-image==0.21.0 vedo==2024.5.1  pyvista==0.43.3  porespy==2.3.0  openpnm==3.4.1 pytrax==0.1.2  pyevtk==1.6.0  pypardiso==0.4.4 && \
  pip cache purge

# artemis: minimum versions that seemed to work natively on Artemis
#RUN pip install numpy==1.20 scipy==1.5.4 pandas==1.1.5  Pillow==8.1.2  imageio==2.9.0  scikit-learn==0.24.2  scikit-image==0.18.1  vedo==2020.3.2  pyvista==0.29.1  porespy==1.2.0 openpnm==2.7.0 pytrax==0.1.2 vtk==9.1.0 && \
# pip cache purge

CMD /bin/bash
