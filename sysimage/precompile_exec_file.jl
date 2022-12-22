import TargetedEstimation

TEST_DIR = joinpath(pkgdir(TargetedEstimation), "test")
push!(LOAD_PATH, TEST_DIR)
cd(TEST_DIR)
include(joinpath(TEST_DIR, "runtests.jl"))