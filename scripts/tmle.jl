using ArgParse
using TargetedEstimation


function parse_commandline()
    s = ArgParseSettings(
        description = "Targeted Learning estimation",
        commands_are_required = false,
        version = "0.2",
        add_version = true)

    @add_arg_table s begin
        "treatments"
            help = "Path to treatment .csv file"
            required = true
        "targets"
            help = "A file (.csv format) containing targets variables (see also --target-type)"
            required = true
        "confounders"
            help = "A file (.csv format) containing the confounding variables values and the sample ids associated"*
                   " with the participants. The first line of the file should contain the columns names and the sample ids "*
                   " column name should be: `SAMPLE_ID`."
            required = true
        "parameters-file"
            help = "A file (.yaml format) see README.md"
            required = true
        "estimator-file"
            help = "A file (.yaml format) describing the tmle estimator to use, README.md"
            required = true
        "out"
            help = "Path where the ouput will be saved"
            required = true
        "--covariates"
            help = "A file (.csv format) containing extra covariates variables for E[Y|X]"
        "--target-type", "-t"
            help = "The type of the target variable: Real or Bool"
            arg_type = String
            default = "Bool"
        "--save-full", "-f"
            help = "Also save the full TMLE estimators for each phenotype."
            action = :store_true
        "--verbosity", "-v"
            help = "Verbosity level"
            arg_type = Int
            default = 1
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

tmle_run(parsed_args)


