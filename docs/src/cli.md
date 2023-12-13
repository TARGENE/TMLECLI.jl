# The Command Line Interface (CLI)

## CLI Installation

### Via Docker (requires Docker)

While we are getting close to providing a standalone application, the most reliable way to use the app is still via the provided [Docker container](https://hub.docker.com/r/olivierlabayle/targeted-estimation/tags). In this container, the command line interface is accessible and can be used directly. For example via:

```bash
docker run -it --rm -v HOST_DIR:CONTAINER_DIR olivierlabayle/targeted-estimation:TAG tmle --help
```

where `HOST_DIR:CONTAINER_DIR` will map the host directory `HOST_DIR` to the container's `CONTAINER_DIR` and `TAG` is the currently released version of the project.

### Build (requires Julia)

Alternatively, provided you have Julia installed, you can build the app via:

```bash
julia --project deps/build_app.jl app
```

Bellow is a description of the functionalities offered by the CLI.

## CLI Description

```@contents
Pages = ["tmle_estimation.md", "sieve_variance.md", "make_summary.md"]
Depth = 5
```
