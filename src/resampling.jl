#####################################################################
#####                      AdativeResampling                    #####
#####################################################################

struct AdaptiveCV <: MLJ.ResamplingStrategy
    shuffle::Bool
    rng::Union{Int, AbstractRNG}
end

"""
    AdaptiveCV(;shuffle=nothing, rng=nothing)

A CV (see `MLJBase.CV`) resampling strategy where the number of folds is determined 
data adaptively based on the rule of thum described [here](https://academic.oup.com/ije/advance-article/doi/10.1093/ije/dyad023/7076266).
"""
AdaptiveCV(;shuffle=nothing, rng=nothing) = AdaptiveCV(MLJBase.shuffle_and_rng(shuffle, rng)...)

base_resampling(::AdaptiveCV) = CV

struct AdaptiveStratifiedCV <: MLJ.ResamplingStrategy
    shuffle::Bool
    rng::Union{Int, AbstractRNG}
end

"""
    AdaptiveStratifiedCV(;shuffle=nothing, rng=nothing)

A StratifiedCV (see `MLJBase.StratifiedCV`) resampling strategy where the number of folds is determined 
data adaptively based on the rule of thum described [here](https://academic.oup.com/ije/advance-article/doi/10.1093/ije/dyad023/7076266).
"""
AdaptiveStratifiedCV(;shuffle=nothing, rng=nothing) = AdaptiveStratifiedCV(MLJBase.shuffle_and_rng(shuffle, rng)...)

base_resampling(::AdaptiveStratifiedCV) = StratifiedCV

AdativeResampling = Union{AdaptiveStratifiedCV, AdaptiveCV}

countuniques(v::AbstractVector) = [count(==(u), v) for u in unique(v)]
countuniques(table) = 
    countuniques([values(x) for x in Tables.namedtupleiterator(table)])

function MLJBase.train_test_pairs(resampling::AdativeResampling, rows, y)
    # Compute n-eff
    n = nrows(y)
    neff = 
        if autotype(y) <: Union{Missing, Continuous}
            n
        else
            counts = countuniques(y)
            nrare = minimum(counts)
            min(n, 5*nrare)
        end

    # Compute number of folds
    nfolds = 
        if neff < 30
            neff
        elseif neff < 500
            20
        elseif neff < 5000
            10
        elseif neff < 10_000
            5
        else
            3
        end
    
    # Constructs base resampling with adapted nfolds
    adapted_cv = base_resampling(resampling)(nfolds=nfolds, shuffle=resampling.shuffle, rng=resampling.rng)
    
    return MLJBase.train_test_pairs(adapted_cv, rows, y)
end


#####################################################################
#####                 JointStratifiedCV                         #####
#####################################################################


mutable struct JointStratifiedCV{T} <: MLJ.ResamplingStrategy where T <: Union{StratifiedCV, AdaptiveStratifiedCV}
    patterns::Union{Nothing, Vector{Regex}}
    resampling::T
end

"""
    JointStratifiedCV(;patterns=nothing, resampling=StratifiedCV())

Applies a stratified cross-validation strategy based on a variable constructed from X and y. 
A composite variable is built from: 
- x variables from X matching any of `patterns` and satisfying `autotype(x) <: Union{Missing, Finite}`. 
If no pattern is provided, then only the second condition is considered.
- y if `autotype(y) <: Union{Missing, Finite}`

The `resampling` needs to be a stratification compliant resampling strategy, at the moment 
one of `StratifiedCV` or `AdaptiveStratifiedCV`
"""
JointStratifiedCV(;patterns=nothing, resampling=StratifiedCV()) = JointStratifiedCV(patterns, resampling)

matches_patterns(colname, patterns::Nothing) = true
matches_patterns(colname, patterns::Vector{Regex}) = any(occursin(p, colname) for p in patterns)

function update_stratification_variable!(stratification_col::AbstractVector, col::AbstractVector)
    for index in eachindex(col)
        stratification_col[index] = join((stratification_col[index], col[index]), "_")
    end
end

function aggretate_finite_col!(stratification_col, col)
    if autotype(col) <: Union{Missing, Finite}
        update_stratification_variable!(stratification_col, col)
    end
end

function aggregate_features!(stratification_col, patterns, X)
    cols = Tables.Columns(X)
    colnames = Tables.columnnames(cols)
    for colname in colnames
        if matches_patterns(string(colname), patterns)
            col = Tables.getcolumn(cols, colname)
            aggretate_finite_col!(stratification_col, col)
        end
    end
end

initialize_aggregate(y) = ["" for _ in eachindex(y)]

function MLJBase.train_test_pairs(resampling::JointStratifiedCV, rows, X, y)
    stratification_col = initialize_aggregate(y)
    aggregate_features!(stratification_col, resampling.patterns, X)
    aggretate_finite_col!(stratification_col, y)
    return MLJBase.train_test_pairs(resampling.resampling, rows, categorical(stratification_col))
end