FROM julia:1.9.4-bullseye

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Europe/Amsterdam

ENV JULIA_DEPOT_PATH=/opt

RUN apt-get update && apt-get install -y wget unzip procps

# Install R and hal9001

RUN apt-get install -y r-base \
                    r-base-core \
                    r-recommended \
                    r-base-dev

RUN apt-get install -y libssl-dev \
                libxml2-dev \
                libcurl4-openssl-dev \
                libgit2-dev \
                libharfbuzz-dev \
                libfribidi-dev \
                libfontconfig1-dev \
                libfreetype6-dev \
                libpng-dev \
                libtiff5-dev \
                libjpeg-dev

RUN R -e "install.packages('devtools', repos='http://cran.us.r-project.org', dependecies=TRUE); \
         require(devtools);\
         install_version('hal9001', version = '0.4.1', repos = 'http://cran.us.r-project.org')"

# Import project, build and precompile

COPY . /TargetedEstimation.jl 

WORKDIR /TargetedEstimation.jl

# Precompile project
RUN julia --project -e'using Pkg; Pkg.instantiate(); Pkg.resolve(); Pkg.precompile()'

# Precompile Sysimage project
RUN julia --project -t auto --startup-file=no deps/build_app.jl

ENV PATH="${PATH}:/TargetedEstimation.jl/build/tmle/bin"

