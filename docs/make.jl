using Documenter
using TmleCLI

DocMeta.setdocmeta!(TmleCLI, :DocTestSetup, :(using TmleCLI); recursive=true)

makedocs(
    authors="Olivier Labayle",
    repo="https://github.com/TARGENE/TmleCLI.jl/blob/{commit}{path}#{line}",
    sitename = "TmleCLI.jl",
    format = Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://TARGENE.github.io/TmleCLI.jl",
        assets=String["assets/logo.ico"],
    ),
    modules = [TmleCLI],
    pages=[
        "Home" => "index.md",
        "Command Line Interface" => ["cli.md", "tmle_estimation.md", "sieve_variance.md", "make_summary.md"],
        "MLJ Extensions" => ["models.md", "resampling.md"],
    ],
    pagesonly=true,
    clean = true,
    checkdocs=:exports
)

@info "Deploying docs..."
deploydocs(;
    repo="github.com/TARGENE/TmleCLI.jl",
    devbranch="main",
    push_preview=true
)