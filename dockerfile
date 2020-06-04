# dlw, june 2020
# to build
# docker build . -t tester
# to test locally
# docker run -it tester:latest


# to run locally run from mpi-sppy

# locally
#start with unix, python, add cplex
#add python dependencies

# on github
# copy in the entire repo
# run the tests


FROM davidwin87/cplex_trial_core_image:latest
ENV PATH $PATH:/opt/ibm/ILOG/CPLEX_Studio_Community129/cplex/bin/x86-64_linux

# for local testing
RUN mkdir /dockerwork
ENV GITHUB_WORKSPACE dockerwork
# end for local testing

RUN apt-get update && apt-get install -y mpich

RUN pip install --upgrade pip
RUN pip install pyomo pandas mpi4py


WORKDIR /dockerwork

COPY ./ ./

RUN python setup.py develop

ENTRYPOINT ["python", "mpisppy/tests/test_ef_ph.py"]