treatmentsfile = joinpath("data", "treatments.csv")
confoundersfile = joinpath("data", "confounders.csv")
iate_param_file = joinpath("config", "iate_parameters.yaml")
ate_param_file = joinpath("config", "ate_parameters.yaml")
continuous_phenotypefile = joinpath("data", "continuous_targets.csv")
binary_phenotypefile = joinpath("data", "binary_targets.csv")
tmle_configfile = joinpath("config", "tmle_config.yaml")
tmle_configfile_2 = joinpath("config", "tmle_config_2.yaml")

function test_queries(queries, expected_queries)
    for (i, query) in enumerate(queries)
        @test query.case == expected_queries[i].case
        @test query.control == expected_queries[i].control
        @test query.name == expected_queries[i].name
    end
end