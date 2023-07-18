using CSV
using DataFrames
using ArgParse
using TargetedEstimation
using TMLE
using Optim
using MLJLinearModels
using Statistics
using MLJBase

function parse_commandline()
    s = ArgParseSettings(
        description = "PheWAS Runtime estimation",
        commands_are_required = false)

    @add_arg_table s begin
        "data"
            help = string("Path to the dataset, a copy is stored on datastore at: ",
                   "/exports/igmm/datastore/ponting-lab/olivier/misc_datasets/sample_ukb_data.csv")
            required = true
            default = "/exports/igmm/datastore/ponting-lab/olivier/misc_datasets/sample_ukb_data.csv"
        "--estimator-file"
            arg_type = String
            help = "Either: glm/glmnet/xgboost/sl"
            required = false
            default = "glm"
    end

    return parse_args(s)
end

function main(parsed_args)
    strat = parsed_args["strategy"]
    parsed_args = Dict(
        "data" => parsed_args["data"],
        "param-file" => "experiments/phewas.param.yaml",
        "estimator-file" => string("experiments/estimator.", strat, ".yaml"),
        "outprefix" => "sample_phewas_output",
        "save-ic" => false,
        "pval-threshold" => 0.05,
        "verbosity" => 0
    )
    t = time()
    tmle_estimation(parsed_args)
    @info string("PheWAS runtime with ", strat, " strategy: ", time() - t)
end

parsed_args = parse_commandline()

main(parsed_args)