module TestSievePlateau

using Test
using DataFrames
using CSV 
using JLD2
using TMLE
using CategoricalArrays
using TargetedEstimation
using StableRNGs
using Distributions
using LogExpFunctions

TESTDIR = joinpath(pkgdir(TargetedEstimation), "test")

function build_dataset(sample_ids)
    rng = StableRNG(123)
    n = size(sample_ids, 1)
    # Confounders
    W₁ = rand(rng, Uniform(), n)
    W₂ = rand(rng, Uniform(), n)
    # Covariates
    C₁ = rand(rng, n)
    # Treatment | Confounders
    T₁ = rand(rng, Uniform(), n) .< logistic.(0.5sin.(W₁) .- 1.5W₂)
    T₂ = rand(rng, Uniform(), n) .< logistic.(-3W₁ - 1.5W₂)
    # target | Confounders, Covariates, Treatments
    μ = 1 .+ 2W₁ .+ 3W₂ .- 4C₁.*T₁ .+ T₁ + T₂.*W₂.*T₁
    y₁ = μ .+ rand(rng, Normal(0, 0.01), n)
    y₂ = rand(rng, Uniform(), n) .< logistic.(μ)
    # Add some missingness
    y₂ = vcat(missing, y₂[2:end])

    dataset = DataFrame(
        SAMPLE_ID = string.(sample_ids),
        T1 = categorical(T₁),
        T2 = categorical(T₂),
        W1 = W₁, 
        W2 = W₂,
        C1 = C₁,
    )

    dataset[!, "CONTINUOUS, OUTCOME"] = y₁
    dataset[!, "BINARY/OUTCOME"] = categorical(y₂)

    CSV.write("data.csv", dataset)
end

function build_tmle_output_file(sample_ids, estimandfile, outprefix, pval)
    build_dataset(sample_ids)
    outputs = TargetedEstimation.Outputs(
        hdf5=TargetedEstimation.HDF5Output(filename=string(outprefix, ".hdf5"), pval_threshold=pval),
    )
    tmle("data.csv", estimandfile, joinpath(TESTDIR, "config", "tmle_ose_config.jl"), outputs=outputs)
end

function basic_variance_implementation(matrix_distance, influence_curve, n_obs)
    variance = 0.f0
    n_samples = size(influence_curve, 1)
    for i in 1:n_samples
        for j in 1:n_samples
            variance += matrix_distance[i, j]*influence_curve[i]* influence_curve[j]
        end
    end
    variance/n_obs
end

function distance_vector_to_matrix!(matrix_distance, vector_distance, n_samples)
    index = 1
    for i in 1:n_samples
        for j in 1:i
            # enforce indicator = 1 when i =j 
            if i == j
                matrix_distance[i, j] = 1
            else
                matrix_distance[i, j] = vector_distance[index]
                matrix_distance[j, i] = vector_distance[index]
            end
            index += 1
        end
    end
end

function test_initial_output(output, expected_output)
    # Metadata columns
    for col in [:PARAMETER_TYPE, :TREATMENTS, :CASE, :CONTROL, :OUTCOME, :CONFOUNDERS, :COVARIATES]
        for index in eachindex(output[!, col])
            if expected_output[index, col] === missing
                @test expected_output[index, col] === output[index, col]
            else
                @test expected_output[index, col] == output[index, col]
            end
        end
    end
end
@testset "Test readGRM" begin
    prefix = joinpath(TESTDIR, "data", "grm", "test.grm")
    GRM, ids = TargetedEstimation.readGRM(prefix)
    @test eltype(ids.SAMPLE_ID) == String
    @test size(GRM, 1) == 18915
    @test size(ids, 1) == 194
end

@testset "Test build_work_list" begin
    grm_ids = TargetedEstimation.GRMIDs(joinpath(TESTDIR, "data", "grm", "test.grm.id"))
    tmpdir = mktempdir(cleanup=true)
    configuration = statistical_estimands_only_config()

    # CASE_1: Since pval = 1.
    # Simulate multiple runs that occured
    pval = 1.
    config_1 = TMLE.Configuration(estimands=configuration.estimands[1:3])
    estimandsfile_1 = joinpath(tmpdir, "configuration_1.json")
    TMLE.write_json(estimandsfile_1, config_1)
    build_tmle_output_file(grm_ids.SAMPLE_ID, estimandsfile_1, "tmle_output_1", pval)

    config_2 = TMLE.Configuration(estimands=configuration.estimands[4:end])
    estimandsfile_2 = joinpath(tmpdir, "configuration_2.json")
    TMLE.write_json(estimandsfile_2, config_2)
    build_tmle_output_file(grm_ids.SAMPLE_ID, estimandsfile_2, "tmle_output_2", pval)

    results, influence_curves, n_obs = TargetedEstimation.build_work_list("tmle_output", grm_ids)
    # Check n_obs
    @test n_obs == [194, 194, 194, 193, 193, 194]
    # Check influence curves
    expected_influence_curves = [size(r.IC, 1) == 194 ? r.IC : vcat(0, r.IC) for r in results]
    for rowindex in 1:6
        @test convert(Vector{Float32}, expected_influence_curves[rowindex]) == influence_curves[rowindex, :]
    end
    # Check results
    all(x isa TMLE.TMLEstimate for x in results)
    all(size(x.IC, 1) > 0 for x in results)
    # clean
    rm("tmle_output_1.hdf5")
    rm("tmle_output_2.hdf5")

    # CASE_2: Since pval = 0.8
    pval = 0.8
    estimandsfile = joinpath(tmpdir, "configuration.json")
    TMLE.write_json(estimandsfile, configuration)
    build_tmle_output_file(grm_ids.SAMPLE_ID, estimandsfile_1, "tmle_output", pval)
    results, influence_curves, n_obs = TargetedEstimation.build_work_list("tmle_output", grm_ids)
    # Check n_obs
    @test n_obs == [194, 194, 194]
    # Check influence curves
    for rowindex in 1:3
        @test convert(Vector{Float32}, results[rowindex].IC) == influence_curves[rowindex, :]
    end
    # Check results
    all(x isa TMLE.TMLEstimate for x in results)
    all(size(x.IC, 1) > 0 for x in results)
    # Clean
    rm("tmle_output.hdf5")
    rm("data.csv")
end

@testset "Test bit_distance" begin
    sample_grm = Float32[-0.6, -0.8, -0.25, -0.3, -0.1, 0.1, 0.7, 0.5, 0.2, 1.]
    nτs = 6
    τs = TargetedEstimation.default_τs(nτs, max_τ=0.75)
    @test τs == Float32[0.0, 0.15, 0.3, 0.45, 0.6, 0.75]
    τs = TargetedEstimation.default_τs(nτs)
    @test τs == Float32[0., 0.4, 0.8, 1.2, 1.6, 2.0]
    d = TargetedEstimation.bit_distances(sample_grm, τs)
    @test d == [0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0
                0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  1.0
                0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
                0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0
                1.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
                1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0]
end

@testset "Test aggregate_variances" begin
    # 2 influence curves containing 5 individuals
    influence_curves = [1. 2. 3. 4. 5.
                        6. 7. 8. 9. 10.]
    # distance indicator with 3 τs and corresponding to row 4
    indicator = [1. 0. 0. 0.2
                 0. 0. 1. 1.
                 1. 0. 1. 1.]
    sample = 4
    var_ = TargetedEstimation.aggregate_variances(influence_curves, indicator, sample)
    @test var_ == [24.0  189.0
                   40.0  225.0
                   48.0  333.0]
end

@testset "Test normalize!" begin
    # 2 τs and 3 curves
    n_obs = [10, 10, 100]
    variances = [1. 2. 3.
                 4. 5. 6.]
    TargetedEstimation.normalize!(variances, n_obs)
    @test variances == [0.1 0.2 0.03
                        0.4 0.5 0.06]
end

@testset "Test compute_variances" begin
    n_curves = 3
    n_samples = 5
    nτs = 5
    n_obs = [3, 4, 4]
    τs = TargetedEstimation.default_τs(nτs)
    # The GRM has 15 lower triangular elements
    grm = Float32[0.4, 0.1, 0.5, 0.2, -0.2, 0.6, 0.3, -0.6, 
                  0.4, 0.3, 0.6, 0.3, 0.7, 0.3, 0.1]
    influence_curves = Float32[0.1 0. 0.1 0.3 0.
                               0.1 0.2 0.1 0.0 0.2
                               0.0 0. 0.1 0.3 0.2]
                  
    
    variances = TargetedEstimation.compute_variances(influence_curves, grm, τs, n_obs)
    @test size(variances) == (nτs, n_curves)

    # when τ=2, all elements are used
    for curve_id in 1:n_curves
        s = sum(influence_curves[curve_id, :])
        var = sum(s*influence_curves[curve_id, i] for i in 1:n_samples)/n_obs[curve_id]
        @test variances[end, curve_id] ≈ var
    end

    # Decreasing variances with τ as all inf curves are positives
    for nτ in 1:nτs-1
        @test all(variances[nτ, :] .<= variances[nτ+1, :])
    end

    # Check against basic_variance_implementation
    matrix_distance = zeros(Float32, n_samples, n_samples)
    for τ_id in 1:nτs
        vector_distance = TargetedEstimation.bit_distances(grm, [τs[τ_id]])
        distance_vector_to_matrix!(matrix_distance, vector_distance, n_samples)
        for curve_id in 1:n_curves
            influence_curve = influence_curves[curve_id, :]
            var_ = basic_variance_implementation(matrix_distance, influence_curve, n_obs[curve_id])
            @test variances[τ_id, curve_id] ≈ var_
        end
    end

    # Check by hand for a single τ=0.5
    @test variances[2, :] ≈ Float32[0.03666667, 0.045, 0.045]
end

@testset "Test grm_rows_bounds" begin
    n_samples = 5
    grm_bounds = TargetedEstimation.grm_rows_bounds(n_samples)
    @test grm_bounds == [1 => 1
                         2 => 3
                         4 => 6
                         7 => 10
                         11 => 15]
end

@testset "Test corrected_stderrors" begin
    io = jldopen(joinpath("data", "sieve_variances.hdf5"))
    variances = io["variances"]
    n_obs = [10, 10, 10, 10, 10, 100, 100, 1000, 1000, 1000]
    stderrors = TargetedEstimation.corrected_stderrors(variances, n_obs)
    # sanity check
    @test size(stderrors, 1) == 10

    # check for the first curve
    stderrors[1] == sqrt(maximum(variances[:,1])/n_obs[1])

    close(io)
end

@testset "Test sieve_variance_plateau" begin
    # Generate data
    nb_estimators = 10
    grm_ids = TargetedEstimation.GRMIDs(joinpath("data", "grm", "test.grm.id"))
    param_file_1 = joinpath("config", "sieve_tests_parameters_1.yaml")
    tmle_outprefix_1 = "tmle_output_1"
    param_file_2 = joinpath("config", "sieve_tests_parameters_2.yaml")
    tmle_outprefix_2 = "tmle_output_2"
    build_tmle_output_file(grm_ids.SAMPLE_ID, param_file_1, tmle_outprefix_1)
    build_tmle_output_file(grm_ids.SAMPLE_ID, param_file_2, tmle_outprefix_2)

    outprefix = "sieve_output"
    parsed_args = Dict(
        "prefix" => "tmle_output",
        "pval" => 1e-10,
        "grm-prefix" => "data/grm/test.grm",
        "out-prefix" => outprefix,
        "nb-estimators" => nb_estimators,
        "max-tau" => 0.75,
        "verbosity" => 0
    )

    sieve_variance_plateau(parsed_args)
    # check hdf5 file
    io = jldopen(string(outprefix, ".hdf5"))
    @test io["taus"] == TargetedEstimation.default_τs(nb_estimators; max_τ=parsed_args["max-tau"])
    @test size(io["variances"]) == (10, 8)
    close(io)
    # check csv file
    output = TargetedEstimation.read_output_with_types(string(outprefix, ".csv"))
    some_expected_cols = DataFrame(
        PARAMETER_TYPE = ["IATE", "IATE", "ATE", "IATE", "IATE", "ATE", "ATE", "CM"],
        TREATMENTS = ["T1_&_T2", "T1_&_T2", "T1_&_T2", "T1_&_T2", "T1_&_T2", "T1_&_T2", "T1", "T1"],
        CASE = ["true_&_true", "true_&_false", "true_&_true", "true_&_true", "true_&_false", "true_&_true", "true", "false"],
        CONTROL = ["false_&_false", "false_&_true", "false_&_false", "false_&_false", "false_&_true", "false_&_false", "false", missing],
        OUTCOME = ["BINARY/OUTCOME", "BINARY/OUTCOME", "BINARY/OUTCOME", "CONTINUOUS, OUTCOME", "CONTINUOUS, OUTCOME", "CONTINUOUS, OUTCOME", "CONTINUOUS, OUTCOME", "CONTINUOUS, OUTCOME"],
        CONFOUNDERS = ["W1_&_W2", "W1_&_W2", "W1_&_W2", "W1_&_W2", "W1_&_W2", "W1_&_W2", "W1", "W1"],
        COVARIATES = ["C1", "C1", "C1", "C1", "C1", "C1", missing, missing]
    )
    test_initial_output(output, some_expected_cols)
    @test output.SIEVE_PVALUE isa Vector{Float64}
    @test output.SIEVE_LWB isa Vector{Float64}
    @test output.SIEVE_UPB isa Vector{Float64}
    @test output.SIEVE_STD isa Vector{Float64}

    tmle_output = TargetedEstimation.load_csv_files(
        TargetedEstimation.empty_tmle_output(),
        ["tmle_output_1.csv", "tmle_output_2.csv"]
    )

    joined = leftjoin(tmle_output, output, on=TargetedEstimation.joining_keys(), matchmissing=:equal)
    @test all(joined.SIEVE_PVALUE .> 0 )
    # clean
    rm(string(outprefix, ".csv"))
    rm(string(outprefix, ".hdf5"))
    rm(string(tmle_outprefix_1, ".hdf5"))
    rm(string(tmle_outprefix_1, ".csv"))
    rm(string(tmle_outprefix_2, ".hdf5"))
    rm(string(tmle_outprefix_2, ".csv"))
    rm("data.csv")
end

end
