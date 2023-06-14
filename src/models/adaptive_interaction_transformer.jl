"""
    RestrictedInteractionTransformer(;order=2, primary_variables=Symbol[], primary_variables_patterns=Regex[])

# Definition

This transformer generates interaction terms based on a set of primary variables. All generated interaction terms 
are composed of a set of primary variables and at most one remaining variable in the provided table. If (T₁, T₂) are
defining the set of primary variables and (W₁, W₂) are reamining variables in the table, the generated interaction terms at order 2 
will be:
- T₁xT₂
- T₁xW₂
- W₁xT₂

but W₁xW₂ will not be generated because it would contain 2 remaining variables.

# Arguments:

- order: All interaction features up to the given order will be computed
- primary_variables: A set of column names to generate the interactions
- primary_variables_patterns: A set of regular expression that can additionally 
be used to identify primary_variables.
"""
MLJModelInterface.@mlj_model mutable struct RestrictedInteractionTransformer <: Unsupervised
    order::Int                                   = 2::(_ > 1)
    primary_variables::Vector{Symbol}            = Symbol[]
    primary_variables_patterns::Vector{Regex}    = Regex[]
end

function retrieve_variables_sets(model::RestrictedInteractionTransformer, X)
    colnames = Tables.columnnames(X)
    primary_from_variables = intersect(colnames, model.primary_variables)
    primary_from_patterns = [colname for colname in colnames 
        if any(occursin(reg, string(colname)) for reg in model.primary_variables_patterns)]
    primary_variables = union(primary_from_variables, primary_from_patterns)
    secondary_variables = setdiff(colnames, primary_variables)
    return primary_variables, secondary_variables
end

InvalidColumnError(colname) = ArgumentError(string("Column ", colname, " does not have infinite scitype and cannot be processed by the RestrictedInteractionTransformer"))

function check_scitypes(X)
    colnames = Tables.columnnames(X)
    for colname in colnames
        col = Tables.getcolumn(X, colname)
        eltype(MLJBase.scitype(col)) <: Infinite || throw(InvalidColumnError(colname))
    end
end

function MLJBase.fit(model::RestrictedInteractionTransformer, verbosity::Int, X)
    check_scitypes(X)
    interactions = Vector{Symbol}[]
    primary_variables, secondary_variables = retrieve_variables_sets(model, X)
    for order in 2:model.order
        # Generate full combinations for the primary set
        for comb in combinations(primary_variables, order)
            push!(interactions, comb)
        end
        # Features from the secondary set should only occur once in each interaction
        for comb in combinations(primary_variables, order - 1)
            for secondary_variable in secondary_variables
                push!(interactions, vcat(comb, secondary_variable))
            end
        end
    end
    return interactions, nothing, nothing
end


interactions(columns, variables...) =
    .*((Tables.getcolumn(columns, var) for var in variables)...)

feature_names(interactions) = 
    Tuple(Symbol(join(inter, "_&_")) for inter in interactions)

function MLJBase.transform(model::RestrictedInteractionTransformer, fitresult, X)
    interaction_features = feature_names(fitresult)
    columns = Tables.Columns(X)
    interaction_table = NamedTuple{interaction_features}([interactions(columns, inter...) for inter in fitresult])
    return merge(Tables.columntable(X), interaction_table)
end

MLJBase.input_scitype(::Type{<:RestrictedInteractionTransformer}) = Table(Continuous)
MLJBase.output_scitype(::Type{<:RestrictedInteractionTransformer}) = Table(Continuous)
