FROM python:latest

MAINTAINER Brian Ritz <bmritz@uchicago.edu>

RUN apt-get update

# install useful system tools and libraries
RUN apt-get install -y libfreetype6-dev && \
    apt-get install -y libglib2.0-0 \
                       libxext6 \
                       libsm6 \
                       libxrender1 \
                       libblas-dev \
                       liblapack-dev \
                       gfortran \
                       libfontconfig1 --fix-missing

RUN pip install matplotlib \
                seaborn \
                pandas \
                numpy \
                scipy \
                sklearn \
                gitpython \
                imbalanced-learn \ 
                jupyter \
                ggplot \
                dask \
                pyyaml \
                python-dateutil \
                h5py

# clone brutils into the site packages
# check if brutils has changed using github public api -- this only breaks cache if there is a new commit
ADD https://api.github.com/repos/bmritz/brutils/compare/master...HEAD /dev/null
RUN git clone https://github.com/bmritz/brutils.git /usr/local/lib/python3.5/site-packages/brutils

# clean up
RUN rm -rf /root/.cache/pip/* && \
    apt-get autoremove -y && apt-get clean
