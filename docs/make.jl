using Documenter
using TargetedEstimation

DocMeta.setdocmeta!(TargetedEstimation, :DocTestSetup, :(using TargetedEstimation); recursive=true)

makedocs(
    authors="Olivier Labayle",
    repo="https://github.com/TARGENE/TargetedEstimation.jl/blob/{commit}{path}#{line}",
    sitename = "TargetedEstimation.jl",
    format = Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://TARGENE.github.io/TargetedEstimation.jl",
        assets=String["assets/logo.ico"],
    ),
    modules = [TargetedEstimation],
    pages=[
        "Home" => "index.md",
        "Command Line Interfaces" => ["environment.md", "tmle_estimation.md", "sieve_variance.md", "merge.md"],
        "MLJ Extensions" => ["models.md", "resampling.md"],
    ]
)

@info "Deploying docs..."
deploydocs(;
    repo="github.com/TARGENE/TargetedEstimation.jl",
    devbranch="main",
    push_preview=true
)