# Run Environment

## General usage

At this point in time, the package depends on several R dependencies which makes it difficult to package as a single Julia executable. We thus rely on a docker container for the execution of the various command line interfaces. 

- This container is available for download from the [docker registry](https://hub.docker.com/r/olivierlabayle/targeted-estimation/tags). 
- In this container, the project is stored in `/TargetedEstimation.jl`, as such, any script can be run using the following template command: `julia --startup-file=no --project=/TargetedEstimation.jl /TargetedEstimation.jl/scripts/SCRIPT_NAME.jl`. Dont forget to mount the output directory in order to retrieve the output data.

## Alternatives

Here are a couple alternatives to using the docker container:

- If you are not using the HAL algorithm, you can simply clone this repository and instantiate the project in order to use the scripts or any other functionality.
- If you are using the HAL algorithm you can use the `docker/Dockerfile` as a guide for your local installation