module TestsUKBB

using Test
using TargetedEstimation
using TMLE
using JLD2


function test_parameters(params, expected_params)
    for (param, expected_param) in zip(params, expected_params)
        @test param.target == expected_param.target
        @test param.treatment == expected_param.treatment
        @test param.confounders == expected_param.confounders
        @test param.covariates == expected_param.covariates
        @test typeof(param) == typeof(expected_param)
    end
end


@testset "Test tmle_run with no extra covariate" begin
    # Only one continuous phenotype / machines not saved / no adaptive cv
    parsed_args = Dict(
        "data" => joinpath("data", "data.csv"),
        "param-file" => joinpath("config", "parameters_no_extra_covariate.yaml"),
        "estimator-file" => joinpath("config", "tmle_config.yaml"),
        "out" => "output.hdf5",
        "verbosity" => 0,
        "no-ic" => false,
    )

    tmle_run(parsed_args)
    
    outfile = jldopen(parsed_args["out"])
    # Parameters are saved only for the first target to save memory
    expected_params = [
        IATE(
            :CONTINUOUS_1, 
            (RSID_10 = (case = "AG", control = "GG"), RSID_100 = (case = "AG", control = "GG")), 
            [:PC1, :PC2], 
            Symbol[]
        ),
        IATE(
            :CONTINUOUS_1, 
            (RSID_10 = (case = "AG", control = "GG"), RSID_100 = (case = "AA", control = "GG")), 
            [:PC1, :PC2], 
            Symbol[]
        ),
        ATE(
            :CONTINUOUS_1, 
            (RSID_10 = (case = "AG", control = "TT"), RSID_100 = (case = "AA", control = "GG")), 
            [:PC1, :PC2], 
            Symbol[]
        )
    ]
    test_parameters(outfile["parameters"], expected_params)
    # results for targets
    var_to_sample_ids = ("CONTINUOUS_1" => 488, "BINARY_1" => 489)
    for (variable, n_samples) in var_to_sample_ids
        results = outfile["results"][variable]
        @test size(results["tmle_results"], 1) == 3
        @test eltype(results["tmle_results"]) == TMLE.AbstractTMLE
        @test size(results["initial_estimates"], 1) == 3 
        @test eltype(results["initial_estimates"]) == Float64
        @test size(results["sample_ids"], 1) == n_samples
    end

    # Clean
    rm(parsed_args["out"])
end


# @testset "Test tmle_run with binary targets" begin
#     parsed_args = Dict(
#         "data" => joinpath("data", "data.arrow"),
#         "param-file" => joinpath("config", "parameters_extra_covariate.yaml"),
#         "estimator-file" => joinpath("config", "tmle_config_2.yaml"),
#         "out" => "output.hdf5",
#         "verbosity" => 1,
#         "no-ic" => false,
#     )

#     tmle_run(parsed_args)
    
#     # Essential results
#     file = jldopen(parsed_args["out"])

#     @test !haskey(file, "SAMPLE_IDS")
    
#     tmlereports = file["TMLEREPORTS"]
#     # Those are summaries not containing the influence curve
#     for key in  ("1_1", "1_2", "2_1", "2_2")
#         @test tmlereports[key].pvalue isa Real
#         @test tmlereports[key].stderror isa Real
#         @test tmlereports[key].confint isa Tuple
#     end

#     machines = file["MACHINES"]
#     Gmach = machines["G"]
#     @test length(report(Gmach).additions.cv_report) == 3
#     Qmach₁ = machines["Q_1"]
#     @test length(report(Qmach₁).cv_report) == 4
#     Qmach₂ = machines["Q_2"]
#     @test length(report(Qmach₂).cv_report) == 4

#     # Clean
#     rm(parsed_args["out"])
# end


end;

true