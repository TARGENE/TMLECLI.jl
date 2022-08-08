#####################################################################
#####                 CV ADAPTIVE FOLDS                          ####
#####################################################################

countuniques(v::AbstractVector) = [count(==(u), v) for u in unique(v)]
countuniques(table) = 
    countuniques([values(x) for x in Tables.namedtupleiterator(table)])

"""
    AdaptiveCV(cv::Union{CV, StratifiedCV})

Implements the rule of thum given here: https://www.youtube.com/watch?v=WYnjja8DKPg&t=4s
"""
mutable struct AdaptiveCV <: MLJBase.ResamplingStrategy
    cv::Union{CV, StratifiedCV}
end


function MLJBase.train_test_pairs(cv::AdaptiveCV, rows, y)
    # Compute n-eff
    n = nrows(y)
    neff = 
        if scitype(first(y)) == MLJBase.Continuous
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
    
    # Update the underlying n_folds
    adapted_cv = typeof(cv.cv)(nfolds=nfolds, shuffle=cv.cv.shuffle, rng=cv.cv.rng)
    
    return MLJBase.train_test_pairs(adapted_cv, rows, y)
end

