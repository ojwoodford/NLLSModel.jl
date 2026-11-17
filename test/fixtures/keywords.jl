const KW_OPTIONS = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=50)

# Every keyword of the set, at a valid value; each entry point takes them all
const KW_SET = (unfixed=nothing, numthreads=static(2), reduction=:none, storage=:auto, max_dense_bytes=10^8, min_work=100, min_work_cost=100)
const KW_ENTRY = (
    ("optimvars", (m; kw...) -> optimvars(m; kw...), KW_SET),
    ("optimvars_nls", (m; kw...) -> optimvars_nls(m; kw...), KW_SET),
    ("flat_optimvars_nls", (m; kw...) -> flat_optimvars_nls(m; kw...), KW_SET),
    ("optimize!, normal", (m; kw...) -> optimize!(deepcopy(m), KW_OPTIONS; kw...), KW_SET),
    ("optimize!, tall", (m; kw...) -> optimize!(deepcopy(m), KW_OPTIONS; formulation=:tall, kw...), KW_SET),
    ("NLPLSModel", (m; kw...) -> NLPLSModel(m; kw...), KW_SET),
    ("NLLSNLSModel", (m; kw...) -> NLLSNLSModel(m; kw...), KW_SET),
)
