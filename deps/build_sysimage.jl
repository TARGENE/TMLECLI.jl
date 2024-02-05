using PackageCompiler
PackageCompiler.create_sysimage(
    ["TargetedEstimation"], 
    cpu_target="generic",
    sysimage_path="TMLESysimage.so", 
    precompile_execution_file="deps/execute.jl", 
)
