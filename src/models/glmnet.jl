mutable struct GLMNetRegressor <: Deterministic
    params::Dict
end

GLMNetRegressor(;params...) = GLMNetRegressor(Dict(params))

mutable struct GLMNetClassifier <: Probabilistic
    params::Dict
end

GLMNetClassifier(;params...) = GLMNetClassifier(Dict(params))

GLMNetModel = Union{GLMNetRegressor, GLMNetClassifier}

make_fitresult(::GLMNetRegressor, res, y) = (glmnetcv=res, )
make_fitresult(::GLMNetClassifier, res, y) = (glmnetcv=res, levels=sort(unique(y)))

MLJBase.reformat(::GLMNetModel, X, y) = (MLJBase.matrix(X), y)
MLJBase.reformat(::GLMNetModel, X) = (MLJBase.matrix(X),)
MLJBase.selectrows(::GLMNetModel, I, Xmatrix, y) = (view(Xmatrix, I, :), view(y, I))
MLJBase.selectrows(::GLMNetModel, I, Xmatrix) = (view(Xmatrix, I, :),)

function MLJBase.fit(model::GLMNetModel, verbosity::Int, Xmatrix, y)
    res = glmnetcv(Xmatrix, y; model.params...)
    return make_fitresult(model, res, y), nothing, nothing
end

MLJBase.predict(::GLMNetRegressor, fitresult, Xmatrix) =
    GLMNet.predict(fitresult.glmnetcv, Xmatrix)


function MLJBase.predict(::GLMNetClassifier, fitresult, Xmatrix)
    raw_probs = GLMNet.predict(fitresult.glmnetcv, Xmatrix, outtype=:prob)
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


function InteractionGLMNetRegressor(;order=2, cache=false, params...)
    return Pipeline(
        interaction_transformer=InteractionTransformer(;order=order),
        glmnet=GLMNetRegressor(;params...),
        cache=cache
    )
end

function InteractionGLMNetClassifier(;order=2, cache=false, params...)
    return Pipeline(
        interaction_transformer=InteractionTransformer(;order=order),
        glmnet=GLMNetClassifier(;params...),
        cache=cache
    )
end