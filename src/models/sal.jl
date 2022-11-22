mutable struct SALRegressor <: MLJ.Deterministic
    evotree
    lasso
    n_iter
end

SALRegressor(;evotree=EvoTreeRegressor(), lasso=LassoRegressor(lambda=0.1), n_iter=3) =
    SALRegressor(evotree, lasso, n_iter)

mutable struct SALClassifier <: MLJ.Probabilistic
    evotree
    lasso
    n_iter
end

SALClassifier(;evotree=EvoTreeClassifier(), lasso=LassoRegressor(), n_iter=10) =
    SALClassifier(evotree, lasso, n_iter)

SAL = Union{SALRegressor, SALClassifier}

iteration_parameter(model::SAL) = :n_iter

function gbt_transform!(H, gbt_machs, X)
    for gbt_index in eachindex(gbt_machs)
        H[:, gbt_index] = MLJBase.predict(gbt_machs[gbt_index], X)
    end
end

function gbt_transform(gbt_machs, X)
    H = Matrix{Float64}(undef, nrows(X), size(gbt_machs, 1))
    gbt_transform!(H, gbt_machs, X)
    return MLJBase.table(H)
end

residuals(model::SALRegressor, y, ŷ) = y .- ŷ

function fit!(gbt_machs::Vector{Machine}, model::SAL, range, X, R, y, verbosity::Int)
    local lasso_mach
    for iter in range
        gbt_machs[iter] = machine(model.evotree, X, R)
        fit!(gbt_machs[iter], verbosity=verbosity)
        H = TargetedEstimation.gbt_transform(gbt_machs[range], X)
        lasso_mach = machine(model.lasso, H, y)
        fit!(lasso_mach, verbosity=verbosity)
        R = TargetedEstimation.residuals(model, y, MLJBase.predict(lasso_mach, H))
    end
    return lasso_mach
end

function update(model::SAL, verbosity, fitresult, cache, X, y)
    current_n_iter = size(fitresult.gbt_machs, 1)
    Δiter = model.n_iter - current_n_iter
    if Δiter > 0
        gbt_machs = Vector{Machine}(undef, model.n_iter)
        gbt_machs[1:current_n_iter] = fitresult.gbt_machs
        H = TargetedEstimation.gbt_transform(fitresult.gbt_machs, X)
        R = TargetedEstimation.residuals(model, y, MLJBase.predict(fitresult.lasso_mach, H))
        range = current_n_iter+1:model.n_iter
        lasso_mach = fit!(gbt_machs, model, range, X, R, y, verbosity)
        return (gbt_machs=gbt_machs, lasso_mach=lasso_mach), nothing, nothing
    end
end

"""
    MLJBase.fit(model::SAL, verbosity::Int, X, y)
"""
function MLJBase.fit(model::SAL, verbosity::Int, X, y)
    gbt_machs = Vector{Machine}(undef, model.n_iter)  
    R = y
    range = 1:model.n_iter
    lasso_mach = TargetedEstimation.fit!(gbt_machs, model, range, X, R, y, verbosity)
    fitresult = (gbt_machs=gbt_machs, lasso_mach=lasso_mach)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MLJBase.predict(model::SAL, fitresult, X)
    H = gbt_transform(fitresult.gbt_machs, X)
    return MLJBase.predict(fitresult.lasso_mach, H)
end

mutable struct EarlyStoppingSALRegressor <: MLJBase.Deterministic
    sal::SALRegressor
    resampling
    epsilon::Float64
end

EarlyStoppingSALRegressor(;sal=SALRegressor(), resampling=Holdout(), epsilon=1e-6) =
    EarlyStoppingSALRegressor(sal, resampling, epsilon=epsilon)



function search_n_iter(model::EarlyStoppingSALRegressor, X, y, epsilon)
    gbt_machs = []
    val_loss = Inf
    iter = 0
    R = y
    while true
        iter += 1
        gbt_machs[iter] = machine(model.sal.evotree, X, R)
        H = gbt_transform(gbt_machs[1:iter], Xs)
        lasso_machs[iter] = machine(:lasso, H, Ys)
        R = TargetedEstimation.residuals(model, Ys, MLJBase.predict(lasso_machs[iter], H))

        if new_val_mse > val_mse  + epsilon
            self.k = len(bases)-1
            return val_mse
        else
            val_mse = new_val_mse
        end
    end
end

"""
This procedure corresponds to a nested early stopping strategy:
    - The outer early stopping loop searches for the best gradient boosting tree and lasso hyperparameters
    - The inner early stopping loop searches for the best number of trees to build
"""
function fit(model::EarlyStoppingSALRegressor, verbosity::Int, X, y)
    val_loss = Inf
    while True
        verbosity >= 1 && @info "Searching for "
        new_val_loss = self.search_n_iter(model, X, y, epsilon=model.epsilon)
        
        if new_val_loss > val_loss + ϵ
            self._set_to(prev_self)
            return self
        else
            val_loss = new_val_loss
            prev_self = copy.deepcopy(self)
            self.gbt.max_depth = model.evotree.max_depth + max_depth_increment
            self.lasso.alpha = model.lasso.alpha * λ_ratio
        end
    end
end

