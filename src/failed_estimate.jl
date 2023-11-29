struct FailedEstimate
    estimand::TMLE.Estimand
    msg::String
end

TMLE.to_dict(x::FailedEstimate) = Dict(
    :estimand => TMLE.to_dict(x.estimand),
    :error => x.msg
)

TMLE.emptyIC(result::FailedEstimate, pval_threshold) = result
