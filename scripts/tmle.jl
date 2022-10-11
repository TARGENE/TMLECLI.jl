using ArgParse
using TargetedEstimation


function parse_commandline()
    s = ArgParseSettings(
        description = "Targeted Learning estimation",
        commands_are_required = false,
        version = "0.2",
        add_version = true)

    @add_arg_table s begin
        "data"
            help = "Path to dataset file (.csv|.arrow)"
            required = true
        "param-file"
            help = "A file (.yaml format) see README.md"
            required = true
        "estimator-file"
            help = "A file (.yaml format) describing the tmle estimator to use, README.md"
            required = true
        "out"
            help = "Path where the ouput will be saved. If `--save-full` is set then"*
                   " this will be a .hdf5 file, otherwise a .csv summary file is output."
            required = true
        "--save-full"
            help = "If the influence curves also need to be stored"
            default = false
            arg_type = Bool
            action = :store_true
        "--verbosity", "-v"
            help = "Verbosity level"
            arg_type = Int
            default = 1
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

main(parsed_args)


