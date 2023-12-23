using PackageCompiler
PackageCompiler.create_app(".", "tmle", 
    lib_name="tmle",
    precompile_execution_file="deps/execute.jl", 
    include_lazy_artifacts=true
)
