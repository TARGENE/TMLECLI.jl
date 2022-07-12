genotypesfile = joinpath("data", "genotypes.csv")
confoundersfile = joinpath("data", "confounders.csv")
iate_queryfile = joinpath("config", "query_iate.toml")
ate_queryfile = joinpath("config", "query_ate.toml")
continuous_phenotypefile = joinpath("data", "continuous_phenotypes.csv")
binary_phenotypefile = joinpath("data", "binary_phenotypes.csv")
phenotypelist_file = joinpath("data", "phen_list.csv")
tmle_configfile = joinpath("config", "tmle_config.yaml")
tmle_configfile_2 = joinpath("config", "tmle_config_2.yaml")

function test_queries(queries, expected_queries)
    for (i, query) in enumerate(queries)
        @test query.case == expected_queries[i].case
        @test query.control == expected_queries[i].control
        @test query.name == expected_queries[i].name
    end
end