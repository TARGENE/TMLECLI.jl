module TargetedEstimation

if occursin("Intel", Sys.cpu_info()[1].model)
    using MKL
end

using DataFrames
using MLJBase
using MLJ
using CSV
using TMLE
using HighlyAdaptiveLasso
using EvoTrees
using MLJLinearModels
using JLD2
using YAML
using CategoricalArrays
using GLMNet
using MLJModels
using Mmap

include("utils.jl")
include("estimators.jl")
include("tmle.jl")
include("sieve_variance.jl")
include("merge.jl")
include(joinpath("models", "glmnet.jl"))
include(joinpath("models", "hal.jl"))


export tmle_estimation, sieve_variance_plateau, merge_csv_files

end
