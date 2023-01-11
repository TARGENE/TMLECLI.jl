using ArgParse
using TargetedEstimation

function parse_commandline()
    s = ArgParseSettings(
        description = "Merge files outputs by tmle.jl and sieve_variance.jl in a single file.",
        commands_are_required = false)

    @add_arg_table s begin
        "tmle-prefix"
            help = "Prefix to files output by tmle.jl"
            required = true
        "out"
            help = "Output file to be generated"
            required = true
        "--sieve-prefix"
            help = "Prefix to files output by sieve_variance.jl"
            required = false
            arg_type = String
    end

    return parse_args(s)
end

parsed_args = parse_commandline()
merge_csv_files(parsed_args)