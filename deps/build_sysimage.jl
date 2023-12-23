using PackageCompiler
PackageCompiler.create_sysimage(
    ["TargetedEstimation"], 
    sysimage_path="TMLESysimage.so", 
    precompile_execution_file="deps/execute.jl", 
)
