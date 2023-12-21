using TargetedEstimation

@info "Running precompilation script."
# Run workload
TEST_DIR = joinpath(pkgdir(TargetedEstimation), "test")
push!(LOAD_PATH, TEST_DIR)
include(joinpath(TEST_DIR, "runtests.jl"))