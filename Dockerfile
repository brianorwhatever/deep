FROM ubuntu:14.04

MAINTAINER <brianorwhatever@gmail.com>

# update and install packages
RUN apt-get update && apt-get --assume-yes upgrade
RUN apt-get -y install wget tmux build-essential gcc g++ make binutils software-properties-common

# download and install nvidia drivers
#RUN  wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
#RUN dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
#RUN apt-get update && apt-get -y upgrade
#RUN apt-get -y install cuda
#RUN modprobe nvidia
#RUN nvidia-smi

# install anaconda
RUN mkdir ~/downloads && cd ~/downloads
RUN wget https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh
RUN bash Anaconda2-4.2.0-Linux-x86_64.sh -b
RUN echo 'export PATH="~/anaconda2/bin:$PATH"' >> ~/.bashrc
RUN export PATH="~/anaconda2/bin:$PATH"
RUN ~/anaconda2/bin/conda install -y bcolz
RUN ~/anaconda2/bin/conda upgrade -y --all

# install theano
RUN ~/anaconda2/bin/pip install theano
RUN echo "[global]\n\
device = gpu\n\
floatX = float32" > ~/.theanorc

# install keras
RUN ~/anaconda2/bin/pip install keras
RUN mkdir ~/.keras
RUN echo '{\n\
    "image_dim_ordering": "th",\n\
    "epsilon": 1e-07,\n\
    "floatx": "float32",\n\
    "backend": "theano"\n\
}' > ~/.keras/keras.json

# install cudnn
#RUN wget http://platform.ai/files/cudnn.tgz
#RUN tar -zxf cudnn.tgz
#RUN cd cuda
#RUN cp lib64/* /usr/local/cuda/lib64/
#RUN cp include/* /usr/local/cuda/include/

# configure jupyter
RUN cd ~/
RUN ~/anaconda2/bin/jupyter notebook --generate-config
#RUN jupass=`~/anaconda2/bin/python -c "from notebook.auth import passwd; print(passwd())"`
#RUN echo "c.NotebookApp.password = u'dl_course'" >> .jupyter/jupyter_notebook_config.py
#RUN echo "c.NotebookApp.ip = '*'\n\
#c.NotebookApp.open_browser = False" >> .jupyter/jupyter_notebook_config.py
RUN mkdir ~/nbs
RUN cd ~/nbs

ADD conf/.tmux.conf /root/

EXPOSE 8888

CMD tmux new -s deep 

#ADD scripts/* /scripts/

#CMD bash
#RUN /scripts/install-gpu.sh

#CMD jupyter notebook
