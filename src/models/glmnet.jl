mutable struct GLMNetRegressor <: Deterministic
    params::Dict
end

GLMNetRegressor(;params...) = GLMNetRegressor(Dict(params))

mutable struct GLMNetClassifier <: Probabilistic
    params::Dict
end

GLMNetClassifier(;params...) = GLMNetClassifier(Dict(params))

GLMNetModel = Union{GLMNetRegressor, GLMNetClassifier}

make_fitresult(model::GLMNetRegressor, res, y) = (glmnetcv=res, )
make_fitresult(model::GLMNetClassifier, res, y) = (glmnetcv=res, levels=sort(unique(y)))

function MLJBase.fit(model::GLMNetModel, verbosity::Int, X, y)
    res = glmnetcv(MLJ.matrix(X), y; model.params...)
    return make_fitresult(model, res, y), nothing, nothing
end

MLJBase.predict(model::GLMNetRegressor, fitresult, X) =
    GLMNet.predict(fitresult.glmnetcv, MLJ.matrix(X))


function MLJBase.predict(model::GLMNetClassifier, fitresult, X)
    raw_probs = GLMNet.predict(fitresult.glmnetcv, MLJ.matrix(X), outtype=:prob)
    levels = fitresult.levels
    if size(levels, 1) == 2
        probs = hcat(1 .- raw_probs, raw_probs)
        preds = UnivariateFinite(levels, probs, pool=missing)
    else
        preds = UnivariateFinite(levels, raw_probs, pool=missing)
    end
    return preds
end

MLJBase.input_scitype(::Type{<:GLMNetModel}) = Table{<:AbstractVector{<:Continuous}}
MLJBase.target_scitype(::Type{<:GLMNetRegressor}) = AbstractVector{<:Continuous}
MLJBase.target_scitype(::Type{<:GLMNetClassifier}) = AbstractVector{<:Finite}