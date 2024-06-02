module Viz

using ..BiasModule: OPES, Metadynamics
using DelimitedFiles
using RecipesBase

export MetaMeasurements, MetaBias, biaspotential, hadroncorrelator, timeseries

# Use Wong colors as default
const default_colors = ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9", "#d55e00"]

"""
    MetaMeasurements(ensemblename::String, fullpath::Bool = false)

Create a `MetaMeasurements` object using all measurement files in the directory `ensemblename`.
Stores measurements taken from files as a `Dict{String, Dict}`,
where the keys of the toplevel Dict are the names of the observables and the sub-Dicts
contain the iterations at which measurements took place and all values.\\
The constructor by default only searches for the directory in the ./measurements folder, but
if you want any directory on your machine to be used, specify `fullpath = true`.
Make sure the directory only contains .txt measurement files produced by MetaQCD.jl or in
the same format.
"""
struct MetaMeasurements
    measurement_dict::Dict{String,Dict{String,Vector{Float64}}}
    observables::Vector{Symbol}
    ensemble::String
    function MetaMeasurements(ensemblename::String)
        dir = pwd() * "/ensembles/$(ensemblename)/measurements"
        hmc_logfile = pwd() * "/ensembles/$(ensemblename)/logs/hmc_acc_logs.txt"
        @assert isdir(dir) "Directory \"$(dir)\" doesn't exist."
        measurement_dict = Dict{String,Dict{String,Vector{Float64}}}()

        filenames = readdir(dir)
        isfile(hmc_logfile) && push!(filenames, hmc_logfile)
        for name in filenames
            name_no_ext = splitext(name)[1]
            measurement = Dict{String,Vector{Float64}}()
            if name == hmc_logfile
                data, header = readdlm(hmc_logfile; header=true)
                for i in eachindex(header)
                    measurement[header[i]] = data[:, i]
                end
                measurement_dict["hmc_data"] = measurement
            elseif occursin("flowed", name_no_ext)
                data, header = readdlm(dir * "/$(name)"; header=true)
                measurement["itrj"] = data[:, 1]
                unique_tflow = unique(data[:, 3])
                unique_indices = Vector{Int64}[]
                for tflow in unique_tflow
                    push!(unique_indices, findall(x -> isapprox(tflow, x), data[:, 3]))
                end

                for i in 4:length(header)
                    for (j, tflow) in enumerate(unique_tflow)
                        measurement[header[i]*" (tf=$(tflow[j]))"] = data[
                            unique_indices[j], i
                        ]
                    end
                end
                measurement_dict[name_no_ext] = measurement
            else
                data, header = readdlm(dir * "/$(name)"; header=true)
                for i in eachindex(header)
                    measurement[header[i]] = data[:, i]
                end
                measurement_dict[name_no_ext] = measurement
            end
        end
        return new(measurement_dict, Symbol.(keys(measurement_dict)), ensemblename)
    end
end

"""
    MetaBias(ensemblename::String; which = nothing, stream::Int = 1, fullpath::Bool = false)

Create a `MetaBias` object using the bias from stream `stream` in the directory `ensemblename`.
This serves as a functor returning the bias value at an input cv. \\
The constructor by default only searches for the directory in the ./metapotentials folder,
but if you want any directory on your machine to be used, specify `fullpath = true`.
Make sure the directory only contains bias files produced by MetaQCD.jl or in
the same format. If the file extension is not .metad or .opes then you will need to
specify `which` as either `:metad` or `:opes`.
"""
struct MetaBias{F}
    bias::F
    function MetaBias(ensemblename::String; which=nothing, stream=1, fullpath=false)
        dir = if fullpath
            ensemblename
        else
            pwd() * "/ensembles/$(ensemblename)/metapotentials/bias/"
        end
        @assert isdir(dir) "Directory \"$(dir)\" doesn't exist."
        filenames = readdir(dir)
        streams = [occursin("stream_$(stream)", name) for name in filenames]
        @assert(
            sum(streams) == 1,
            "There has to be exactly 1 file pertaining to stream $(stream) in the directory."
        )
        file = dir * filenames[findfirst(x -> x == true, streams)]
        ext = splitext(file)[end]

        if ext == ".metad" || which == :metad
            data = readdlm(file; skipstart=1)
            cvlims = data[1, 1], data[end, 1]
            bin_width = data[2, 1] - data[1, 1]
            bin_vals = data[:, 1]
            values = data[:, 2]
            bias = Metadynamics(true, 1, cvlims, Inf, bin_width, 1.0, 100, bin_vals, values)
        elseif ext == ".opes" || which == :opes
            bias = OPES(file)
        else
            throw(AssertionError("File extension $ext not recognized.
                                 Must be either .metad or .opes"))
        end

        return new{typeof(bias)}(bias)
    end
end

(m::MetaBias{F})(cv::Float64) where {F} = m.bias(cv)

function Base.show(io::IO, m::MetaMeasurements)
    print(io, "MetaMeasurements(")
    print(io, "\"$(m.ensemble)\"")
    print(io, ")")
    return nothing
end

function Base.show(io::IO, m::MetaBias)
    print(io, "MetaBias(")
    print(io, typeof(m.bias))
    print(io, ")")
    return nothing
end

function Base.getproperty(m::MetaMeasurements, s::Symbol)
    s == :ensemble && return getfield(m, :ensemble)
    s == :measurement_dict && return getfield(m, :measurement_dict)
    s == :observables && return getfield(m, :observables)
    valid_names = Symbol.(keys(m.measurement_dict))
    @assert s ∈ valid_names "Your MetaMeasurements don't contain the observable $s"
    return getfield(m, :measurement_dict)["$(s)"]
end

observables(m::MetaMeasurements) = m.observables

@userplot TimeSeries

"""
    timeseries(m::MetaMeasurements, observable::Symbol)

Plot the time series of the observable `observable` from the measurements in `m`.
"""
RecipesBase.@recipe function timeseries(ts::TimeSeries; seriestype=:line)
    m, observable = ts.args[1:2]
    @assert !occursin("correlator", string(observable)) "timeseries not supported for correlators"
    @assert observable ∈ observables(m) "Observable $observable is not in Measurements"
    seriestype := seriestype
    obs_keys = collect(keys(getproperty(m, observable)))
    filter!(x -> x ≠ "itrj", obs_keys)
    palette --> default_colors
    x = try
        getproperty(m, observable)["itrj"]
    catch _
        nothing
    end

    if occursin("bias_data", string(observable))
        size --> (600, 200 * length(obs_keys))
        cv = getproperty(m, observable)["cv"]
        filter!(x -> x ≠ "cv", obs_keys)
        link := :x
        layout := (length(obs_keys) + 1, 1)
        legend := false

        @series begin
            xlabel --> ""
            ylabel --> "cv"
            subplot := 1
            y = cv
            x, y
        end

        for (i, name) in enumerate(obs_keys)
            @series begin
                xl = i == length(obs_keys) ? "Monte Carlo Time" : ""
                xlabel --> xl
                ylabel --> name
                color --> default_colors[i+1]
                subplot := i + 1
                y = getproperty(m, observable)[name]
                x, y
            end
        end
    elseif occursin("hmc_data", string(observable))
        xlabel --> "Monte Carlo Time"
        ylabel --> "$(observable)"

        # for name in obs_keys
        @series begin
            label --> "ΔH"
            y = getproperty(m, observable)["ΔH"]
            11:length(y), y[11:end]
        end
        # end
    else
        xlabel --> "Monte Carlo Time"
        ylabel --> "$(observable)"

        for name in obs_keys
            @series begin
                label --> name
                y = getproperty(m, observable)[name]
                x, y
            end
        end
    end
end

@userplot BiasPotential

"""
    biaspotential(bias::MetaBias; cvlims::NTuple{2, AbstractFloat})

Plot the bias potential `bias` in the range given by either the cvlims of the bias itself
or `cvlims` if specified. If the cvlims of the bias contain infinities and `cvlims` is not
specified then the limits default to [-5, 5].
"""
RecipesBase.@recipe function biaspotential(bp::BiasPotential; cvlims=nothing)
    b = bp.args[1]
    bias = b.bias
    xlims = cvlims ≡ nothing ? bias.cvlims : cvlims
    isinf(sum(xlims)) && (xlims = (-5, 5))

    legend := false
    palette --> default_colors
    xlabel --> "Collective Variable"
    ylabel --> "Bias Potential ($(typeof(bias)))"
    x = (bias isa OPES) ? (xlims[1]:0.001:xlims[2]-0.001) : bias.bin_vals
    y = b.(x)
    return x, y
end

@userplot HadronCorrelator

"""
    hadroncorrelator(m::MetaMeasurements, correlator::Symbol)

Plot the effective mass plot of the hadron correlator `correlator` from the measurements in `m`.
"""
RecipesBase.@recipe function hadroncorrelator(
    hc::HadronCorrelator; logscale=false, style=:line, tf=0, calc_meff=false
)
    m, correlator = hc.args[1:2]
    tmp = tf > 0 ? :_correlator_flowed : :_correlator
    observable = Symbol(correlator, tmp)
    @assert observable ∈ observables(m) "Observable $observable is not in Measurements"
    seriestype := style
    obs_keys = collect(keys(getproperty(m, observable)))
    filter!(x -> x ≠ "itrj", obs_keys)
    filter!(x -> x ≠ "C" && x ≠ "C_flowed", obs_keys)
    palette --> default_colors
    x = collect(1:length(obs_keys))
    len = length(x)
    C = zeros(len)
    Cr = zeros(len)
    tmp = last.(split.(obs_keys, "_"))
    if tf > 0
        tmp = split.(tmp, " ")
        tmp = [tmp[i][1] for i in eachindex(tmp)]
    end
    nums = parse.(Int, tmp)

    str(it) = tf > 0 ? "$(correlator)_corr_$(it) (tf=$tf)" : "$(correlator)_corr_$(it)"
    for it in nums
        tmp = getproperty(m, observable)[str(it)]
        C[it] = sum(tmp) / length(tmp)
    end
    key_str = tf > 0 ? "C_flowed" : "C"
    haskey(getproperty(m, observable), key_str) || (getproperty(m, observable)[key_str] = C)
    if calc_meff
        for it in nums
            Cr[it] = log(C[it] / C[mod1(it + 1, len)])
        end
    end

    xlabel --> "Time Extent"
    ylabel --> "⟨C(t)⟩"
    xticks := 1:len
    yscale := logscale ? :log10 : :identity
    linecolor := default_colors[1]
    markercolor := default_colors[1]
    markershape := :circ

    @series begin
        label --> string(observable)
        y = calc_meff ? Cr : C
        x, y
    end
end

end
