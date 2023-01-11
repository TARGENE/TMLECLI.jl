
function build_formula_hal(X, column_pattern)
    formula = "~h(.)"
    for name in Tables.columnnames(X)
        if occursin(column_pattern, string(name))
            formula *= " + h($name, .)"
        end
    end
    return formula
end

struct SNPInteractionHALClassifier <: MLJ.Probabilistic
    column_pattern
    hal::HALClassifier
end

SNPInteractionHALClassifier(;column_pattern="^rs[0-9]+", kwargs...) =
    SNPInteractionHALClassifier(Regex(column_pattern), HALClassifier(;kwargs...))

struct SNPInteractionHALRegressor <: MLJ.Deterministic
    column_pattern
    hal::HALRegressor
end

SNPInteractionHALRegressor(;column_pattern="^rs[0-9]+", kwargs...) =
    SNPInteractionHALRegressor(Regex(column_pattern), HALRegressor(;kwargs...))

SNPInteractionHAL = Union{SNPInteractionHALRegressor,SNPInteractionHALClassifier}

MLJ.target_scitype(model::SNPInteractionHALRegressor) = AbstractVector{<:MLJ.Continuous}
MLJ.target_scitype(model::SNPInteractionHALClassifier) = AbstractVector{<:Finite}
MLJ.input_scitype(model::SNPInteractionHAL) = Table

function MLJ.fit(model::SNPInteractionHAL, verbosity::Int, X, y)
    formula_hal = build_formula_hal(X, model.column_pattern)
    model.hal.formula = formula_hal
    return MLJ.fit(model.hal, verbosity, X, y)
end

MLJ.predict(model::SNPInteractionHAL, fitresult, X) =
    MLJ.predict(model.hal, fitresult, X)