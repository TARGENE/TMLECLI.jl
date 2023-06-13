using Documenter
using TargetedEstimation

makedocs(
    sitename = "TargetedEstimation",
    format = Documenter.HTML(),
    modules = [TargetedEstimation]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
