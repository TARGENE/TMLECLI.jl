using PackageCompiler
PackageCompiler.create_app(".", "tmle", precompile_execution_file="deps/execute.jl", include_lazy_artifacts=true)
