FROM julia:1.9.4-bullseye

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get -y install gcc mono-mcs && \
    rm -rf /var/lib/apt/lists/*

ENV TZ=Europe/Amsterdam

ENV JULIA_DEPOT_PATH=/opt

# Import project, build and precompile

COPY . /TargetedEstimation.jl 

WORKDIR /TargetedEstimation.jl

# Precompile project
RUN julia --project -e'using Pkg; Pkg.instantiate(); Pkg.resolve(); Pkg.precompile()'

# Precompile Sysimage project
# RUN julia --project -t auto --startup-file=no deps/build_app.jl app

# ENV PATH="${PATH}:/TargetedEstimation.jl/build/tmle/bin"

