FROM ubuntu:latest

MAINTAINER <brianorwhatever@gmail.com>

# update apt and install packages
RUN apt-get update
RUN apt-get install -y tmux

CMD tmux new -s deep
