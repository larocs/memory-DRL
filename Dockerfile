FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install wget xz-utils python3 python3-pip git libglib2.0-0 xvfb -y

RUN wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_2_0_Ubuntu20_04.tar.xz && \
    tar -xvf /CoppeliaSim_Edu_V4_2_0_Ubuntu20_04.tar.xz

ENV COPPELIASIM_ROOT=/CoppeliaSim_Edu_V4_2_0_Ubuntu20_04/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
ENV QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

RUN mkdir /mysac
COPY ./ /mysac/

ADD id_rsa /root/.ssh/id_rsa
RUN git clone git@gitlab.com:LaRoCS/PyRep.git \
    --single-branch updated_pyrep_with_marta
RUN pip3 install -r PyRep/requirements.txt
RUN pip3 install -e /PyRep/
RUN pip3 install -e /mysac/

WORKDIR /mysac/