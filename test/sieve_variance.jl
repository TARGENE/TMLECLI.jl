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

include(joinpath(TESTDIR, "testutils.jl"))

function write_sieve_dataset(sample_ids)
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
    dataset[!, "COUNT_OUTCOME"] = rand(rng, [1, 2, 3, 4], n)

    CSV.write("data.csv", dataset)
end

function build_tmle_output_file(sample_ids, estimandfile, outprefix; 
    pval=1., 
    estimatorfile=joinpath(TESTDIR, "config", "tmle_ose_config.jl")
    )
    write_sieve_dataset(sample_ids)
    outputs = TargetedEstimation.Outputs(
        hdf5=TargetedEstimation.HDF5Output(filename=string(outprefix, ".hdf5"), pval_threshold=pval, sample_ids=true),
    )
    tmle("data.csv"; 
        estimands=estimandfile, 
        estimators=estimatorfile, 
        outputs=outputs
    )
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

    # CASE_1: pval = 1.
    # Simulate multiple runs that occured
    config_1 = TMLE.Configuration(estimands=configuration.estimands[1:3])
    estimandsfile_1 = joinpath(tmpdir, "configuration_1.json")
    TMLE.write_json(estimandsfile_1, config_1)
    build_tmle_output_file(grm_ids.SAMPLE_ID, estimandsfile_1, "tmle_output_1")

    config_2 = TMLE.Configuration(estimands=configuration.estimands[4:end])
    estimandsfile_2 = joinpath(tmpdir, "configuration_2.json")
    TMLE.write_json(estimandsfile_2, config_2)
    build_tmle_output_file(grm_ids.SAMPLE_ID, estimandsfile_2, "tmle_output_2")

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

    # CASE_2: pval = 0.1
    pval = 0.1
    estimandsfile = joinpath(tmpdir, "configuration.json")
    TMLE.write_json(estimandsfile, configuration)
    build_tmle_output_file(grm_ids.SAMPLE_ID, estimandsfile, "tmle_output"; pval=pval)
    results, influence_curves, n_obs = TargetedEstimation.build_work_list("tmle_output", grm_ids)
    # Check n_obs
    @test n_obs == [194, 193, 193, 194]
    # Check influence curves
    expected_influence_curves = [size(r.IC, 1) == 194 ? r.IC : vcat(0, r.IC) for r in results]
    for rowindex in 1:4
        @test convert(Vector{Float32}, expected_influence_curves[rowindex]) == influence_curves[rowindex, :]
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
    variances = [
        1. 2. 6.
        4. 5. 3.
    ]
    stderrors = TargetedEstimation.corrected_stderrors(variances)
    # sanity check
    @test stderrors == sqrt.([4., 5., 6.])
end

@testset "Test SVP" begin
    # Generate data
    grm_ids = TargetedEstimation.GRMIDs(joinpath(TESTDIR, "data", "grm", "test.grm.id"))
    tmpdir = mktempdir(cleanup=true)
    configuration = statistical_estimands_only_config()
    pval = 0.1
    config_1 = TMLE.Configuration(estimands=configuration.estimands[1:3])
    estimandsfile_1 = joinpath(tmpdir, "configuration_1.json")
    TMLE.write_json(estimandsfile_1, config_1)
    build_tmle_output_file(grm_ids.SAMPLE_ID, estimandsfile_1, "tmle_output_1"; pval=pval)

    config_2 = TMLE.Configuration(estimands=configuration.estimands[4:end])
    estimandsfile_2 = joinpath(tmpdir, "configuration_2.json")
    TMLE.write_json(estimandsfile_2, config_2)
    build_tmle_output_file(grm_ids.SAMPLE_ID, estimandsfile_2, "tmle_output_2"; pval=pval)

    # Using the main command
    main([
        "svp", 
        "tmle_output", 
        "--grm-prefix", joinpath(TESTDIR, "data", "grm", "test.grm"), 
        "--max-tau", "0.75"
    ])

    io = jldopen("svp.hdf5")
    # Check τs
    @test io["taus"] == TargetedEstimation.default_τs(10; max_τ=0.75)
    # Check variances
    @test size(io["variances"]) == (10, 4)
    # Check results
    svp_results = io["results"]
    
    tmleout1 = jldopen(x -> x["Batch_1"], "tmle_output_1.hdf5")
    tmleout2 = jldopen(x -> x["Batch_1"], "tmle_output_2.hdf5")
    src_results = [tmleout1..., tmleout2...]

    for svp_result in svp_results
        src_result_index = findall(x.TMLE.estimand == svp_result.TMLE.estimand for x in src_results)
        src_result = src_results[only(src_result_index)]
        @test src_result.TMLE.std != svp_result.TMLE.std
        @test src_result.TMLE.estimate == svp_result.TMLE.estimate
        @test src_result.TMLE.n == svp_result.TMLE.n
        @test svp_result.TMLE.IC == []
    end

    close(io)
    # clean
    rm("svp.hdf5")
    rm("tmle_output_1.hdf5")
    rm("tmle_output_2.hdf5")
    rm("data.csv")
end

@testset "Test SVP: causal and composed estimands" begin
    # Generate data
    grm_ids = TargetedEstimation.GRMIDs(joinpath(TESTDIR, "data", "grm", "test.grm.id"))
    tmpdir = mktempdir(cleanup=true)
    configuration = causal_and_joint_estimands_config()
    pval = 1.
    configfile = joinpath(tmpdir, "configuration.json")
    TMLE.write_json(configfile, configuration)
    build_tmle_output_file(
        grm_ids.SAMPLE_ID, 
        configfile, 
        "tmle_output";
        estimatorfile=joinpath(TESTDIR, "config", "ose_config.jl")
    )

    # Using the main command
    main([
        "svp", 
        "tmle_output", 
        "--grm-prefix", joinpath(TESTDIR, "data", "grm", "test.grm"), 
        "--max-tau", "0.75",
        "--estimator-key", "OSE"
    ])

    # The JointEstimate std is not updated but each component is.
    src_results = jldopen(x -> x["Batch_1"], "tmle_output.hdf5")
    io = jldopen("svp.hdf5")
    svp_results = io["results"]
    standalone_estimates = svp_results[1:2]
    from_composite = svp_results[3:4]
    @test standalone_estimates[1].OSE.estimand == from_composite[1].OSE.estimand
    @test standalone_estimates[2].OSE.estimand == from_composite[2].OSE.estimand

    # Check std has been updated
    for i in 1:2
        @test standalone_estimates[i].OSE.estimand == src_results[i].OSE.estimand
        @test standalone_estimates[i].OSE.estimate == src_results[i].OSE.estimate
        @test standalone_estimates[i].OSE.std != src_results[i].OSE.std
    end

    close(io)
    
    # clean
    rm("svp.hdf5")
    rm("tmle_output.hdf5")
    rm("data.csv")
end

end

true
