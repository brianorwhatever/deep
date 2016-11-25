alias ssh-p2='ssh -i ~/.ssh/deep-p2.pem ubuntu@ec2-35-160-13-55.us-west-2.compute.amazonaws.com'
alias deep-build='docker build -t deep .'
alias deep-run='nvidia-docker run -it -p 8888:8888 --net=host -v /home/brian/deep/nbs:/nbs deep'
