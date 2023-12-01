using TargetedEstimation

@info "Running precompilation script."

# Run help messages
TargetedEstimation.command_main(["-h"])
TargetedEstimation.command_main(["tmle", "-h"])
TargetedEstimation.command_main(["make-summary", "-h"])
TargetedEstimation.command_main(["sieve-variance-plateau", "-h"])

# Run workload
TEST_DIR = joinpath(pkgdir(TargetedEstimation), "test")
push!(LOAD_PATH, TEST_DIR)
include(joinpath(TEST_DIR, "runtests.jl"))