# The Command Line Interface

## Installing the CLI

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

Be low is a description of the functionalities offered by the CLI.

## CLI Description

The CLI contains 3 sub-commands:

- `tmle`: To Run TMLE on a dataset (see [tmle command](@ref)).
- `sieve-variance-plateau`: To correct the variance of an estimator for non i.i.d data via [Sieve Variance Plateau](https://biostats.bepress.com/ucbbiostat/paper322/) (see [sieve-variance-plateau command](@ref)).
- `make-summary`: Combines multiple outputs from a `tmle` run into one output file (see [make-summary command](@ref))

### tmle command

Arguments:

- dataset: A dataset either in .csv or .arrow format
- estimands: A file containing a serialized Configuration object.
- estimators: A custom julia file containing the estimators to use. Several examples are provided [here](https://github.com/TARGENE/TargetedEstimation.jl/estimators-configs). Alternatively, to point to any of them, the name of the file can be supplied without the ".jl" extension. (e.g. "superlearning").

Options:

- -v, --verbosity: Verbosity level.
- -o, --outputs: Ouputs to be generated.
- --chunksize <100::Int>: Results are written in batches of size chunksize.
- -r, --rng <123::Int>: Random seed (Only used for estimands ordering at the moment).
- -c, --cache-strategy: Caching Strategy for the nuisance functions, any of ("release-unusable", "no-cache", "max-size").

Flags:

- -s, --sort-estimands: Sort estimands to minimize cache usage. A brute force approach will be used, resulting in exponentially long sorting time (Only appropriate for small number of estimands).