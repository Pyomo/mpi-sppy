# for weekly tests == dlw, june 2020
# docker build . -t mpisppyweekly:latest
# to test locally:
# docker run -it mpisppyweekly:latest
# docker run -v /home/woodruff/Documents/Research/mpi-sppy/:/mpi-sppy -it mpisppyweekly:latest
# (see also the cell2fire Dockerfile)
# docker tag c2fcondatest dlwoodruff/mpisppyweekly:latest
# docker push dlwoodruff/mpisppyweekly:latest
##############################################
FROM continuumio/anaconda3
RUN conda update conda
RUN conda install -c anaconda numpy
RUN conda install -c anaconda pandas

RUN apt-get update && apt-get install -y mpich

RUN pip install --upgrade pip
RUN pip install  mpi4py
RUN pip install xpress

RUN apt update
RUN apt install -y git-all

# get pyutilib and pyomo from the web
RUN git clone https://github.com/PyUtilib/pyutilib.git
RUN cd pyutilib && python ./setup.py develop && cd ..
# (the last cd .. is probably not needed)

RUN git clone https://github.com/pyomo/pyomo
RUN cd pyomo && git remote add bknueven https://github.com/bknueven/pyomo && git fetch bknueven && git checkout -b benxpress bknueven/xpress_solver

RUN cd pyomo && python setup.py develop
