using Documenter
using NLLSModel

DocMeta.setdocmeta!(NLLSModel, :DocTestSetup, :(using NLLSModel); recursive=true)

makedocs(
    sitename = "NLLSModel.jl",
    modules = [NLLSModel],
    checkdocs = :exports,
    remotes = nothing,
    warnonly = false,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        sidebar_sitename = true,
    ),
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
        "Tutorials & Guide" => [
            "Defining Residuals & Variables" => "tutorials/defining_residuals.md",
            "Robust Cost Functions" => "tutorials/robust_loss.md",
            "Linear Solvers & Sparsity" => "tutorials/linear_solvers.md",
            "Bundle Adjustment & Alternating Minimization" => "tutorials/bundle_adjustment.md",
            "NLPModels.jl Integration" => "tutorials/nlpmodels.md",
        ],
        "Migration Guides" => [
            "From NLLSsolver.jl" => "migration/from_nllssolver.md",
            "From Ceres Solver (C++)" => "migration/from_ceres.md",
        ],
    ]
)

deploydocs(
    repo = "github.com/ojwoodford/NLLSModel.jl.git",
    devbranch = "main",
)
