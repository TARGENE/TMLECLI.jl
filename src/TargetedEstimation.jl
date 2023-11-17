module TargetedEstimation

if occursin("Intel", Sys.cpu_info()[1].model)
    using MKL
end

using DataFrames
using MLJBase
using MLJ
using CSV
using Arrow
using TMLE
using HighlyAdaptiveLasso
using EvoTrees
using MLJXGBoostInterface
using MLJLinearModels
using JLD2
using CategoricalArrays
using GLMNet
using MLJModels
using Mmap
using Serialization
using MultipleTesting
using Combinatorics
using Tables
using Random
using YAML

import MLJModelInterface

include("cache_managers.jl")
include("runner.jl")
include("utils.jl")
include("sieve_variance.jl")
include("merge.jl")
include("resampling.jl")
include(joinpath("models", "glmnet.jl"))
include(joinpath("models", "adaptive_interaction_transformer.jl"))
include(joinpath("models", "biallelic_snp_encoder.jl"))

export run_estimation, sieve_variance_plateau, merge_csv_files
export GLMNetRegressor, GLMNetClassifier
export RestrictedInteractionTransformer, BiAllelicSNPEncoder
export AdaptiveCV, AdaptiveStratifiedCV, JointStratifiedCV

end
