using PackageCompiler
PackageCompiler.create_app(".", "tmle",
    executables = ["tmle" => "julia_main"],
    precompile_execution_file="deps/execute.jl", 
    include_lazy_artifacts=true
)
