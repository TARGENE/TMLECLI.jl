using TmleCLI

@info "Running precompilation script."
# Run workload
TEST_DIR = joinpath(pkgdir(TmleCLI), "test")
push!(LOAD_PATH, TEST_DIR)
include(joinpath(TEST_DIR, "runtests.jl"))