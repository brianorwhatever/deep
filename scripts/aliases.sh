alias deep-build='docker build -t deep .'
alias deep-run='nvidia-docker run -it -p 8888:8888 --net=host -v /home/brian/deep/nbs:/nbs deep'
