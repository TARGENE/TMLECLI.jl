using ArgParse
using TargetedEstimation

function parse_commandline()
    s = ArgParseSettings(
        description = "Targeted Learning Estimation",
        commands_are_required = false,
        version = "0.2",
        add_version = true)

    @add_arg_table s begin
        "dataset"
            help = "Path to dataset file (.csv|.arrow)"
            required = true
        "estimands-config"
            help = "A .yaml file listing all parameters to estimate."
            required = true
        "--estimators-config"
            help = "A file (.jl) defining the estimators to be used."
            arg_type= String
            required = false
        "--hdf5-out"
            help = "Stores the results in a HDF5 file format (see also: --pval-threshold)."
            arg_type = String
            default = nothing
        "--csv-out"
            help = "Path to an output `.csv` file."
            required = true
        "--pval-threshold"
            help = """In order to save disk space, only estimation results with a p-value lesser than 
            the threshold will have their influence curve saved. (default = 1., i.e. all influence curves are saved).
            """
            default = 1.
            arg_type = Float64
        "--sort-estimands"
            help = "If estimands should be sorted to minimize memory usage, see also: cache-strategy."
            default = false 
            arg_type = Bool
        "--cache-strategy"
            help = string("Nuisance functions are stored in the cache during estimation. The cache can be released from these",
            " functions to limit memory consumption. There are currently 3 caching management strategies: ",
            "'release_unusable' (default): Will release the cache from nuisance functions that won't be used in the future. ",
            "'K': Will keep the cache size under K nuisance functions. ",
            "'no_cache': Disables caching. ",
            "Note that caching strategies are better used in conjunction with `--sort-estimands` to minimized memory usage."
            )
            default = "release_unusable"
            arg_type = String
        "--chunksize"
            help = "Results are appended to outfiles in chunks."
            default = 100
            arg_type = Int
        "--rng"
            help = "Random seed"
            default = 123
            arg_type = Int
        "--verbosity", "-v"
            help = "Verbosity level"
            arg_type = Int
            default = 1
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

tmle_estimation(parsed_args)


