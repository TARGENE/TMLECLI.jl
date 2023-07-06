
"""
    BiAllelicSNPEncoder(patterns=Symbol[])

Encodes bi-allelic SNP columns, identified by the provided `patterns` Regex, 
as a count of a reference allele determined dynamically (not necessarily the minor allele).
"""
mutable struct BiAllelicSNPEncoder <: Unsupervised 
    patterns::Vector{Regex}
    BiAllelicSNPEncoder(patterns) = new([x isa Regex ? x : Regex(x) for x in patterns])
end

BiAllelicSNPEncoder(;patterns=[]) = BiAllelicSNPEncoder(patterns)

isok_eltype(::Type{<:CategoricalValue{String,}}) = true
isok_eltype(::Type{<:Union{Missing, <:CategoricalValue{String,}}}) = true
isok_eltype(v) = false

function MLJModelInterface.fit(model::BiAllelicSNPEncoder, verbosity::Int, X)
    columns = Tables.Columns(X)
    ref_alleles = Dict{Symbol, Char}()
    for colname in Tables.columnnames(columns)
        if any(occursin(pattern, string(colname)) for pattern in model.patterns)
            column = Tables.getcolumn(columns, colname)
            isok_eltype(eltype(column)) || throw(NonCategoricalVectorError(colname))
            val = column[findfirst(x -> x !== missing, column)]
            genotypes = levels(val)
            # This operation will return Chars
            alleles = unique(Iterators.flatten(genotypes))
            # Check only two alleles
            length(alleles) == 2 || throw(NonBiallelicSNPError(colname))
            for genotype in genotypes
                # Check genotypes are bi-allelic
                length(genotype) == 2 || throw(NonBiAllelicGenotypeError(colname, genotype))
            end
            ref_alleles[colname] = alleles[1]
        end
    end
    return ref_alleles, nothing, nothing
end

function count_nref!(newcolumn, column, ref_allele)
    for index in eachindex(column)
        val = column[index]
        if val !== missing
            newcolumn[index] = count(x -> x === ref_allele, unwrap(val))
        end
    end
end

function MLJModelInterface.transform(model::BiAllelicSNPEncoder, fitresult, X)
    columns = Tables.Columns(X)
    colnames = Tuple(Tables.columnnames(columns))
    snpcolnames = keys(fitresult)
    newcolumns = AbstractVector[]
    for colname in colnames
        column = Tables.getcolumn(columns, colname)
        if colname ∈ snpcolnames
            ref_allele = fitresult[colname]
            newcoltype = Missing <: eltype(column) ? Union{Missing, Int} : Int
            newcolumn = Vector{newcoltype}(undef, size(column, 1))
            TargetedEstimation.count_nref!(newcolumn, column, ref_allele)
        else
            newcolumn = column
        end
        push!(newcolumns, newcolumn)
    end
    return NamedTuple{colnames}(newcolumns)
end

MLJModelInterface.input_scitype(::Type{<:BiAllelicSNPEncoder}) = Table
MLJModelInterface.output_scitype(::Type{<:BiAllelicSNPEncoder}) = Table

NonCategoricalVectorError(colname) = ArgumentError(string("Column ", colname, " matches the bi-allelic SNP pattern but is not a CategoricalVector. Please convert first.")) 
NonBiallelicSNPError(colname) = ArgumentError(string(colname, " does not correspond to a bi-allelic SNP."))
NonBiAllelicGenotypeError(colname, genotype) = ArgumentError(string("Genotype: ", genotype, ", in column: ", colname, " does not correspond to a bi-allelic SNP."))