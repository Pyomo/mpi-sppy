# dlw, june 2020
# to build locally:
# docker build . -t tester
# to test locally:
# docker run -it tester:latest

FROM davidwin87/cplex_trial_core_image:latest
ENV PATH $PATH:/opt/ibm/ILOG/CPLEX_Studio_Community129/cplex/bin/x86-64_linux

# for local testing
###RUN mkdir /dockerwork
###ENV GITHUB_WORKSPACE dockerwork
# end for local testing

RUN apt-get update && apt-get install -y mpich

RUN pip install --upgrade pip
RUN pip install pyomo pandas mpi4py

WORKDIR /dockerwork

COPY ./ ./

RUN python setup.py develop

ENTRYPOINT ["python", "mpisppy/tests/test_ef_ph.py"]