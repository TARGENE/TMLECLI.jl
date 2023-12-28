FROM julia:1.10-bullseye

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get -y install gcc mono-mcs vim procps && \
    rm -rf /var/lib/apt/lists/*

ENV TZ=Europe/Amsterdam

ENV JULIA_DEPOT_PATH=/opt

# Import the project

COPY . /TargetedEstimation.jl 

WORKDIR /TargetedEstimation.jl

# Precompile the project
RUN julia --project -e'using Pkg; Pkg.instantiate(); Pkg.resolve(); Pkg.precompile()'

# Build CLI
RUN julia --project --startup-file=no deps/build_sysimage.jl

# Test the CLI runs
RUN julia --startup-file=no --project -JTMLESysimage.so tmle.jl tmle data/sample_dataset.csv