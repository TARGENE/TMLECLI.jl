using TargetedEstimation
using Documenter

DocMeta.setdocmeta!(TargetedEstimation, :DocTestSetup, :(using TargetedEstimation); recursive=true)

makedocs(;
    modules=[TargetedEstimation],
    authors="Olivier Labayle",
    repo="https://github.com/olivierlabayle/TargetedEstimation.jl/blob/{commit}{path}#{line}",
    sitename="TargetedEstimation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://olivierlabayle.github.io/TargetedEstimation.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/olivierlabayle/TargetedEstimation.jl",
    devbranch="main",
)
