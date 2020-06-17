# NOT USED BY GITHUB
# dlw, june 2020
# to build locally:
# docker build . -t tester
# to test locally:
# docker run -it tester:latest
# docker run -v /home/woodruff/Documents/Research/mpi-sppy/:/mpi-sppy -it tester:latest

FROM davidwin87/cplex_trial_core_image:latest
ENV PATH $PATH:/opt/ibm/ILOG/CPLEX_Studio_Community129/cplex/bin/x86-64_linux

RUN apt-get update && apt-get install -y mpich

RUN pip install --upgrade pip
RUN pip install pyomo pandas mpi4py

# assume mpisppy is mounted
# cd to it and run setup.py
