using MetaQCD: OPES, Metadynamics
using DelimitedFiles
using Plots; gr()
using RecipesBase
using Roots
using SingularSpectrumAnalysis

# Use Wong colors as default
const default_colors = [
    RGBA{Float32}(0.0f0,0.44705883f0,0.69803923f0,1.0f0),
    RGBA{Float32}(0.9019608f0,0.62352943f0,0.0f0,1.0f0),
    RGBA{Float32}(0.0f0,0.61960787f0,0.4509804f0,1.0f0),
    RGBA{Float32}(0.8f0,0.4745098f0,0.654902f0,1.0f0),
    RGBA{Float32}(0.3372549f0,0.7058824f0,0.9137255f0,1.0f0),
    RGBA{Float32}(0.8352941f0,0.36862746f0,0.0f0,1.0f0),
    RGBA{Float32}(0.9411765f0,0.89411765f0,0.25882354f0,1.0f0),
]

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
    measurement_dict::Dict{String, Dict{String, Vector{Float64}}}
    observables::Vector{Symbol}
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
end

(m::MetaBias{F})(cv::Float64) where {F} = m.bias(cv)

function Base.show(io::IO, m::MetaMeasurements)
    print(io, "MetaMeasurements(")
    print(io, keys(m.measurement_dict))
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
    s == :measurement_dict && return getfield(m , :measurement_dict)
    s == :observables && return getfield(m , :observables)
    valid_names = Symbol.(keys(m.measurement_dict))
    @assert s ∈ valid_names "Your MetaMeasurements don't contain the observable $s"
    return getfield(m, :measurement_dict)["$(s)"]
end

observables(m::MetaMeasurements) = m.observables

function MetaMeasurements(ensemblename::String)
    dir = pwd()*"/ensembles/$(ensemblename)/measurements"
    hmc_logfile = pwd()*"/ensembles/$(ensemblename)/logs/hmc_acc_logs.txt"
    @assert isdir(dir) "Directory \"$(dir)\" doesn't exist."
    measurement_dict = Dict{String, Dict{String, Vector{Float64}}}()

    filenames = readdir(dir)
    isfile(hmc_logfile) && push!(filenames, hmc_logfile)
    for name in filenames
        name_no_ext = splitext(name)[1]
        measurement = Dict{String, Vector{Float64}}()
        if name == hmc_logfile
            data, header = readdlm(hmc_logfile; header=true)
            for i in eachindex(header)
                measurement[header[i]] = data[:, i]
            end
            measurement_dict["hmc_data"] = measurement
        elseif occursin("flowed", name_no_ext)
            data, header = readdlm(dir*"/$(name)"; header=true)
            measurement["itrj"] = data[:, 1]
            unique_tflow = unique(data[:, 3])
            unique_indices = Vector{Int64}[]
            for tflow in unique_tflow
                push!(unique_indices, findall(x->isapprox(tflow, x), data[:, 3]))
            end

            for i in 4:length(header)
                for (j, tflow) in enumerate(unique_tflow)
                    measurement[header[i]*" (tf=$(tflow[j]))"] = data[unique_indices[j], i]
                end
            end
            measurement_dict[name_no_ext] = measurement
        else
            data, header = readdlm(dir*"/$(name)"; header=true)
            for i in eachindex(header)
                measurement[header[i]] = data[:, i]
            end
            measurement_dict[name_no_ext] = measurement
        end
    end
    return MetaMeasurements(measurement_dict, Symbol.(keys(measurement_dict)))
end

function MetaBias(ensemblename::String; which=nothing, stream=1, fullpath=false)
    dir = fullpath ? ensemblename : pwd()*"/ensembles/$(ensemblename)/metapotentials/bias/"
    @assert isdir(dir) "Directory \"$(dir)\" doesn't exist."
    filenames = readdir(dir)
    streams = [occursin("stream_$(stream)", name) for name in filenames]
    @assert(sum(streams)==1,
        "There has to be exactly 1 file pertaining to stream $(stream) in the directory.")
    file = dir * filenames[findfirst(x->x==true, streams)]
    ext = splitext(file)[end]

    if ext==".metad" || which==:metad
        data = readdlm(file, skipstart=1)
        cvlims = data[1, 1], data[end, 1]
        bin_width = data[2, 1] - data[1, 1]
        bin_vals = data[:, 1]
        values = data[:, 2]
        bias = Metadynamics(true, 1, cvlims, Inf, bin_width, 1.0, 100, bin_vals, values)
    elseif ext==".opes" || which==:opes
        bias = OPES(file)
    else
        throw(AssertionError("File extension $ext not recognized.
                             Must be either .metad or .opes"))
    end

    return MetaBias(bias)
end

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
    filter!(x->x≠"itrj", obs_keys)
    palette --> default_colors
    x = try
        getproperty(m, observable)["itrj"]
    catch _
        nothing
    end

    if occursin("bias_data", string(observable))
        size --> (600, 200*length(obs_keys))
        cv = getproperty(m, observable)["cv"]
        filter!(x->x≠"cv", obs_keys)
        link := :x
        layout := (length(obs_keys)+1, 1)
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
                xl = i==length(obs_keys) ? "Monte Carlo Time" : ""
                xlabel --> xl
                ylabel --> name
                color --> default_colors[i+1]
                subplot := i+1
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
    xlims = cvlims≡nothing ? bias.cvlims : cvlims
    isinf(sum(xlims)) && (xlims = (-5, 5))

    legend := false
    palette --> default_colors
    xlabel --> "Collective Variable"
    ylabel --> "Bias Potential ($(typeof(bias)))"
    x = (bias isa OPES) ? (xlims[1]:0.001:xlims[2]-0.001) : bias.bin_vals
    y = b.(x)
    x, y
end

@userplot HadronCorrelator

"""
    hadroncorrelator(m::MetaMeasurements, correlator::Symbol)

Plot the effective mass plot of the hadron correlator `correlator` from the measurements in `m`.
"""
RecipesBase.@recipe function hadroncorrelator(hc::HadronCorrelator; style=:line)
    m, correlator = hc.args[1:2]
    observable = Symbol(correlator, :_correlator) 
    @assert observable ∈ observables(m) "Observable $observable is not in Measurements"
    seriestype := style
    obs_keys = collect(keys(getproperty(m, observable)))
    filter!(x->x≠"itrj", obs_keys)
    palette --> default_colors
    x = collect(1:length(obs_keys))
    len = length(x)
    C = zeros(len)
    nums = parse.(Int, last.(split.(obs_keys, "_")))

    for it in nums
        tmp = getproperty(m, observable)["$(correlator)_corr_$(it)"]
        C[it] = sum(tmp) / length(tmp)  
    end

    xlabel --> "Time Extent"
    ylabel --> "⟨C(t)⟩"
    xticks := 1:len
    linecolor := default_colors[1]
    markercolor := default_colors[1]
    markershape := :circ

    @series begin
        label --> string(observable)
        x, C
    end
end

function create_modified(b::MetaBias)
    # TODO: Modify the bias and then create a .metad file in the same directory
    # containing the modified bias.
end
