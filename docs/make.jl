using Documenter
using NLLSModels
using BlockMatrices
using OptimizationVariableInterface

DocMeta.setdocmeta!(NLLSModels, :DocTestSetup, :(using NLLSModels); recursive=true)

makedocs(
    sitename = "NLLSModels.jl",
    modules = [NLLSModels, BlockMatrices, OptimizationVariableInterface],
    checkdocs = :none,
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
        ],
        "Migration Guides" => [
            "From NLLSsolver.jl" => "migration/from_nllssolver.md",
            "From Ceres Solver (C++)" => "migration/from_ceres.md",
        ],
    ]
)

deploydocs(
    repo = "github.com/ojwoodford/NLLSModels.jl.git",
    devbranch = "main",
)
