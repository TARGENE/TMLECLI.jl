#####################################################################
###                       CONSTANT BASIS                          ###
#####################################################################

mutable struct ConstantBasis{T<:AbstractFloat} <: Deterministic
    value::T
end

ConstantBasis(;value=0.) = ConstantBasis(value)

MLJ.fit(model::ConstantBasis, verbosity::Int, X, y) = nothing, nothing, nothing

MLJ.predict(model::ConstantBasis, fitresult, X) = fill(model.value, nrows(X))

#####################################################################
###                          BASIC SAL                            ###
#####################################################################

mutable struct SALFitresult
    constant
    gbts
    lasso
    SALFitresult() = new()
end

mutable struct SALRegressor <: MLJ.Deterministic
    gbt::EvoTreeRegressor
    lasso::LassoRegressor
    n_bases::Int
end

SALRegressor(;gbt=EvoTreeRegressor(nrounds=100), lasso=LassoRegressor(lambda=1.), n_bases=10) =
    SALRegressor(gbt, lasso, n_bases)

mutable struct SALClassifier <: MLJ.Probabilistic
    gbt::EvoTreeClassifier
    lasso::LogisticClassifier
    n_bases::Int
end

SALClassifier(;gbt=EvoTreeClassifier(nrounds=100), lasso=LassoRegressor(lambda=1.), n_bases=10) =
    SALClassifier(gbt, lasso, n_bases)

SAL = Union{SALRegressor, SALClassifier}

function gbt_transform!(H, gbts, constant, X)
    for gbt_index in eachindex(gbts)
        H[:, gbt_index] = MLJBase.predict(gbts[gbt_index], X)
    end
    H[:, end] = MLJBase.predict(constant, X)
end

function transform(fitresult::SALFitresult, X, iter)
    gbts = fitresult.gbts[1:iter]
    H = Matrix{Float64}(undef, nrows(X), size(gbts, 1) + 1)
    transform!(H, gbts, fitresult.constant, X)
    return MLJBase.table(H)
end

constant_basis(model::SALRegressor, y) = ConstantBasis(value=mean(y))

initialize_gradient(model::SALRegressor, n) = zeros(n)

update_gradient!(∇::AbstractVector, model::SALRegressor, y, ŷ) = y .- ŷ

function fit_new_basis!(fitresult::SALFitresult, model, X, y, iter, verbosity)
    # Initialize basis with a constant prediction and array of gradient boosting trees
    if iter == 0
        fitresult.constant = machine(constant_basis(model, y), X, y)
        fit!(fitresult.constant, verbosity=verbosity)
        fitresult.gbts = Vector{Machine}(undef, model.n_bases)
    # Fit new gradient boosting tree
    else
        fitresult.gbts[iter] = machine(model.gbt, X, y)
        fit!(fitresult.gbts[iter], verbosity=verbosity)
    end
end

function fit_lasso!(fitresult::SALFitresult, verbosity::Int, X, y)
    fitresult.lasso = machine(model.lasso, H, y)
    MLJBase.fit!(lasso_mach, verbosity=verbosity)
end

"""
    MLJBase.fit(model::SAL, verbosity::Int, X, y)
"""
function MLJBase.fit(model::SAL, verbosity::Int, X, y)
    fitresult = SALFitresult()
    fit_new_basis!(fitresult, model, X, y, 0, verbosity)
    ∇ = initialize_gradient(model, nrows(y))
    for iter in model.n_bases
        ∇ = TargetedEstimation.update_gradient!(∇, model, fitresult, y)
        fit_new_basis!(fitresult, model, X, ∇, iter, verbosity)
        H = TargetedEstimation.transform(fitresult, X, iter)
        lasso_mach = machine(model.lasso, H, y)
        MLJBase.fit!(lasso_mach, verbosity=verbosity)
        R = TargetedEstimation.residuals(model, y, MLJBase.predict(lasso_mach, H))
    end
    lasso_mach = TargetedEstimation.update!(bases, model, X, y, verbosity)
    fitresult = (gbt_machs=gbt_machs, lasso_mach=lasso_mach)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MLJBase.predict(model::SAL, fitresult, X)
    H = gbt_transform(fitresult.gbt_machs, X)
    return MLJBase.predict(fitresult.lasso_mach, H)
end

#####################################################################
###                      EARLY STOPPING SAL                       ###
#####################################################################

mutable struct EarlyStoppingSALRegressor <: Deterministic
    sal::SAL
    resampling::ResamplingStrategy
    λ_ratio::Float64
    depth_increment::Int
    ϵ::Float64
    max_bases::Int
end

EarlyStoppingSALRegressor(;sal=SALRegressor(), resampling=Holdout(), λ_ratio=0.1, depth_increment=1, ϵ=1e-6, max_bases=100) =
    EarlyStoppingSALRegressor(sal, resampling, λ_ratio, depth_increment, ϵ, max_bases)

stopping_criterion(model::EarlyStoppingSALRegressor, new_loss, old_loss) = 
    abs(new_loss - old_loss) < model.ϵ

function select_bases!(model, R, gbt_machs, train_test_pairs, X, y; verbosity=0)
    val_loss = Inf
    for n_bases in 1:model.max_bases
        cv_losses = Vector{Float64}(undef, length(train_test_pairs))
        for (cvid, (train, val)) in enumerate(train_test_pairs)
            Xtr = selectrows(X, train)
            Rtr = R[cvid]
            ytr = selectrows(y, train)
            gbt_mach = machine(model.sal.gbt, Xtr, Rtr)
            fit!(gbt_mach, verbosity=verbosity-1)
            push!(gbt_machs[cvid], gbt_mach)
            Htr = TargetedEstimation.gbt_transform(gbt_machs[cvid][1:n_bases], Xtr)
            lasso_mach = machine(model.sal.lasso, Htr, ytr)
            fit!(lasso_mach, verbosity=verbosity-1)
            ŷtr = MLJ.predict(lasso_mach)
            R[cvid] = TargetedEstimation.residuals(model.sal, ytr, ŷtr)
            ŷval = MLJ.predict(
                lasso_mach,
                TargetedEstimation.gbt_transform(gbt_machs[cvid][1:n_bases], selectrows(X, val))
            )

            cv_losses[cvid] = TargetedEstimation.loss(model.sal, ŷval, selectrows(y, val))
        end

        new_val_loss = mean(cv_losses)

        # Probably rework the stopping criterion
        Δloss = abs(new_val_loss - val_loss)
        verbosity > 1 && @info "Δ loss: $Δloss"
        do_stop = stopping_criterion(model, new_val_loss, val_loss)
        val_loss = new_val_loss
        if do_stop
            model.sal.n_bases = n_bases
            return val_loss
        end
    end

    @warn "Inner early stopping loop failed to reach stopping point."
    return val_loss
end

function select_hyperparameters(model, X, y; verbosity=0)
    sal_template = model.sal
    model.sal = deepcopy(model.sal)
    best_sal = deepcopy(model.sal)
    train_test_pairs = MLJBase.train_test_pairs(model.resampling, 1:nrows(y), X, y)
    R = TargetedEstimation.initialize_residuals(model.sal, y, train_test_pairs)
    gbt_machs = [Machine[] for _ in 1:length(train_test_pairs)]
    val_loss = Inf
    i = 1
    while true
        new_val_loss = TargetedEstimation.select_bases!(model, R, gbt_machs, train_test_pairs, X, y; verbosity=verbosity)
        verbosity > 0 && @info "SAL validation loss at hyperparameter iteration $i : $new_val_loss"
        i += 1
        # Stopping criterion and return should be rethought
        if new_val_loss > val_loss + model.ϵ
            model.sal = sal_template
            return best_sal
        else
            val_loss = new_val_loss
            verbosity > 0 && @info "Updating Lasso & GBT hyperparameters."
            best_sal = deepcopy(model.sal)
            model.sal.lasso.lambda *= model.λ_ratio
            model.sal.gbt.max_depth += model.depth_increment
        end
    end
end

function MLJBase.fit(model::EarlyStoppingSALRegressor, verbosity::Int, X, y)
    sal = select_hyperparameters(model, X, y; verbosity=verbosity)
    mach = machine(sal, X, y)
    fit!(mach, verbosity=verbosity-1)
    return mach, nothing, nothing
end

function MLJBase.predict(model::EarlyStoppingSALRegressor, fitresult, X)
    MLJBase.predict(fitresult, X)
end

loss(sal::SALRegressor, ŷ, y) = root_mean_squared_error(ŷ, y)

function initialize_residuals(sal::SALRegressor, y, train_test_pairs) 
    R = Vector{Vector{Float64}}(undef, length(train_test_pairs))
    for (i, (train, _)) in enumerate(train_test_pairs)
        R[i] = selectrows(y, train)
    end
    return R
end





