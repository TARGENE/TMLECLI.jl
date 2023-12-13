using ArgParse
using TargetedEstimation

const ESTIMATORS = [
    "glm",
    "glmnet",
    "G-superlearning-Q-glmnet",
    "superlearning"
]
const PARAMETERS = [
    "experiments/parameters.phewas.yaml",
    "experiments/parameters.continuous.gwas.yaml",
    "experiments/parameters.binary.gwas.yaml"
]

function parse_commandline()
    s = ArgParseSettings(
        description = "Runs TMLE for 100 SNPs, 1 binary and one continuous trait. Runtime will depend on the platform.",
        commands_are_required = false)

    @add_arg_table s begin
        "data"
            help = string("Path to the dataset, a copy is stored on datastore at: ",
                   "/exports/igmm/datastore/ponting-lab/olivier/misc_datasets/sample_ukb_data.csv")
            required = true
            default = "/exports/igmm/datastore/ponting-lab/olivier/misc_datasets/sample_ukb_data.csv"
        "--estimator-file"
            arg_type = String
            help = "Any estimator file from: docs/src/estimators/"
        "--param-file"
            arg_type = String
            help = """
            Any parameter file from: 
            experiments/parameters.phewas.yaml
            experiments/parameters.binary.gwas.yaml
            experiments/parameters.continuous.gwas.yaml
            """
        "--verbosity"
            arg_type = Int
            help = "Verbosity level"
            default = 0
            required = false
    end

    return parse_args(s)
end


function main(parsed_args)
    param_files =  parsed_args["param-file"] isa Nothing ? PARAMETERS : [parsed_args["param-file"]]
    estimator_files = parsed_args["estimator-file"] isa Nothing ? ESTIMATORS : [parsed_args["estimator-file"]]
    for (paramfile, estimatorfile) in Iterators.product(param_files, estimator_files)
        tmle_args = Dict(
            "data" => parsed_args["data"],
            "param-file" => paramfile,
            "estimator-file" => estimatorfile,
            "csv-out" => "runtime_output.csv",
            "hdf5-out" => nothing,
            "pval-threshold" => 0.05,
            "chunksize" => 100,
            "verbosity" => parsed_args["verbosity"],
        )
        nparams = length(TargetedEstimation.parameters_from_yaml(paramfile))

        # Time it: this will include precompilation time
        t_start = time()
        tmle_estimation(tmle_args)
        t_end = time()

        totaltime = t_end - t_start
        unittime = totaltime / nparams

        summary_info = 
        """
        TMLE was run with:
        - parameter file: $paramfile.
        - estimatorfile: $estimatorfile.

        Total time was: $totaltime seconds.
        Unit time was: $unittime seconds.
        """
        @info summary_info
    end
end

parsed_args = parse_commandline()

main(parsed_args)

