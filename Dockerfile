FROM ubuntu:14.04

MAINTAINER <brianorwhatever@gmail.com>

RUN apt-get update && apt-get install -y sudo
ADD scripts/* /scripts/

CMD bash
#RUN /scripts/install-gpu.sh

#CMD tmux new -s deep
