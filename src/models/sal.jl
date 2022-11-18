struct SALRegressor <: MLJ.DeterministicNetworkComposite
    evotree
    lasso
    n_iter
end

SALRegressor(;evotree=EvoTreeRegressor(), lasso=LassoRegressor(), n_iter=10) =
    SALRegressor(evotree, lasso, n_iter)

struct SALClassifier <: MLJ.ProbabilisticNetworkComposite
    evotree
    lasso
    n_iter
end

SALClassifier(;evotree=EvoTreeClassifier(), lasso=LassoRegressor(), n_iter=10) =
    SALClassifier(evotree, lasso, n_iter)

SAL = Union{SALRegressor, SALClassifier}


function gbt_transform!(H, gbt_machs, Xs)
    for gbt_index in eachindex(gbt_machs)
        H[gbt_index] = MLJBase.predict(gbt_machs[gbt_index], Xs)
    end
end

function gbt_transform(gbt_machs, Xs)
    H = Vector{Node}(undef, size(gbt_machs, 1))
    gbt_transform!(H, gbt_machs, Xs)
    return MLJBase.table(hcat(H...))
end

residuals(model::SALRegressor, y, ŷ) = y .- ŷ

residuals(model::SALRegressor, y::AbstractNode, ŷ::AbstractNode) = 
    node((y, ŷ) -> residuals(model, y, ŷ), y, ŷ)

"""
    MLJBase.prefit(model::SAL, verbosity::Int, X, y)
"""
function MLJBase.prefit(model::SAL, verbosity::Int, X, y)
    Xs = source(X)
    Ys = source(y)
    gbt_machs = Vector{Machine}(undef, model.n_iter)
    lasso_machs = Vector{Machine}(undef, model.n_iter)
    R = Ys
    
    for iter in 1:model.n_iter
        gbt_machs[iter] = machine(:evotree, Xs, R)
        H = gbt_transform(gbt_machs[1:iter], Xs)
        lasso_machs[iter] = machine(:lasso, H, Ys)
        R = TargetedEstimation.residuals(model, Ys, MLJBase.predict(lasso_machs[iter], H))
    end

    ŷ = MLJBase.predict(
        lasso_machs[end], 
        TargetedEstimation.gbt_transform(gbt_machs, Xs)
    )

    return (predict=ŷ,)
end