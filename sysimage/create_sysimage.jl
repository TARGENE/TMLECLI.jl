
SYSIMAGE_DIR = dirname(@__FILE__)
push!(LOAD_PATH, SYSIMAGE_DIR)

using PackageCompiler

create_sysimage(
    ["TargetedEstimation"]; 
    sysimage_path="TargetedEstimationSysimage.so",
    precompile_execution_file=joinpath(SYSIMAGE_DIR, "precompile_exec_file.jl")
)