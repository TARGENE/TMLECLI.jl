using Documenter
using TMLECLI

DocMeta.setdocmeta!(TMLECLI, :DocTestSetup, :(using TMLECLI); recursive=true)

makedocs(
    authors="Olivier Labayle",
    repo="https://github.com/TARGENE/TMLECLI.jl/blob/{commit}{path}#{line}",
    sitename = "TMLECLI.jl",
    format = Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://TARGENE.github.io/TMLECLI.jl",
        assets=String["assets/logo.ico"],
    ),
    modules = [TMLECLI],
    pages=[
        "Home" => "index.md",
        "Command Line Interface" => ["cli.md", "tmle_estimation.md", "make_summary.md"],
        "MLJ Extensions" => ["models.md", "resampling.md"],
    ],
    pagesonly=true,
    clean = true,
    checkdocs=:exports
)

@info "Deploying docs..."
deploydocs(;
    repo="github.com/TARGENE/TMLECLI.jl",
    devbranch="main",
    push_preview=true
)