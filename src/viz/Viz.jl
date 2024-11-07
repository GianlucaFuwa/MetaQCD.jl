module Viz

using ..BiasModule: OPES, Metadynamics
using DelimitedFiles
using RecipesBase

include("endpointranges.jl")

export MetaMeasurements, MetaBias, biaspotential, eigenvalues, hadroncorrelator, timeseries
export ibegin, iend

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
struct MetaMeasurements # TODO: overload Base.show and do some @level1 printing
    measurement_dict::Dict{String,Dict{String,Vector{Float64}}}
    observables::Vector{Symbol}
    ensemble::String
    function MetaMeasurements(ensemblename::String; fullpath=false)
        dir = if fullpath
            ensemblename
        else
            pkgdir(Viz) * "/ensembles/$(ensemblename)/measurements"
        end

        hmc_logfile = pkgdir(Viz) * "/ensembles/$(ensemblename)/logs/hmc_acc_logs.txt"
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
                unique_tflow = unique(data[:, 3])
                unique_indices = Vector{Int64}[]
                measurement["itrj"] = data[1:length(unique_tflow):end, 1]
                
                for tflow in unique_tflow
                    push!(unique_indices, findall(x -> isapprox(tflow, x), data[:, 3]))
                end

                for i in 4:length(header)
                    for (j, tflow) in enumerate(unique_tflow)
                        measurement[header[i]*" (tf=$(tflow))"] = data[
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

Base.length(m::MetaMeasurements, observable) = Int(getproperty(m, observable)["itrj"][end])

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
    ensemblename::String
    function MetaBias(ensemblename::String; which=nothing, stream=1, fullpath=false)
        dir = if fullpath
            ensemblename
        else
            pkgdir(Viz) * "/ensembles/$(ensemblename)/metapotentials/"
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
            bias = Metadynamics(
                true,
                1,
                cvlims,
                Inf,
                bin_width,
                1.0,
                100,
                bin_vals,
                values,
            )
        elseif ext == ".opes" || which == :opes
            bias = OPES(file)
        else
            throw(AssertionError("File extension $ext not recognized.
                                 Must be either .metad or .opes"))
        end

        return new{typeof(bias)}(bias, ensemblename)
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
timeseries(m::MetaMeasurements, observable::Symbol, seriestype=:line, irange=Colon())

Plot the time series of the observable `observable` from the measurements in `m` in the
between the iteration specified by `irange`. If you want to, e.g., plot from iteration
50 till the end, do `irange=50:iend`
"""
RecipesBase.@recipe function timeseries(
    ts::TimeSeries; seriestype=:line, irange=Colon(), tf=nothing,
)
    m, observable = ts.args[1:2]
    @assert !occursin("correlator", string(observable)) "timeseries not supported for correlators"
    @assert !occursin("eigenvalues", string(observable)) "timeseries not supported for eigenvalues"
    @assert observable ∈ observables(m) "Observable $observable is not in Measurements"
    seriestype := seriestype
    obs_keys = collect(keys(getproperty(m, observable)))
    filter!(x -> x ≠ "itrj", obs_keys)
    palette --> default_colors

    x = try
        view(getproperty(m, observable)["itrj"], irange)
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
        palette --> default_colors

        @series begin
            xlabel --> ""
            ylabel --> "cv"
            yticks --> floor(minimum(cv)):ceil(maximum(cv))
            subplot := 1
            y = view(cv, irange)
            x, y
        end

        for (i, name) in enumerate(obs_keys)
            @series begin
                xl = i == length(obs_keys) ? "Monte Carlo Time" : ""
                xlabel --> xl
                ylabel --> name
                color --> default_colors[i+1]
                subplot := i + 1
                y = view(getproperty(m, observable)[name], irange)
                x, y
            end
        end
    elseif occursin("hmc_data", string(observable))
        palette --> default_colors
        xlabel --> "Monte Carlo Time"
        ylabel --> "$(observable)"

        # for name in obs_keys
        @series begin
            label --> "ΔH"
            y = view(getproperty(m, observable)["ΔH"], irange)
            x, y
        end
        # end
    elseif occursin("flowed", string(observable))
        # size --> (600, 250 * length(obs_keys))
        palette --> default_colors
        xlabel --> "Monte Carlo Time"
        ylabel --> first(split(obs_keys[1], " "))
        linewidth --> 2
        legend --> :outertopright
        # layout := (length(obs_keys), 1)
        nlabel = last.(split.(obs_keys, " "))
        tf_digits = parse.(Float64, filter.(x -> isdigit(x) || x=='.', nlabel))
        if tf === nothing 
            iordered = sortperm(tf_digits)        
        else
            iordered = findall(x -> x==tf, tf_digits)
        end

        for (j, i) in enumerate(iordered)
            @series begin
                # subplot := j
                label --> nlabel[i]
                y = view(getproperty(m, observable)[obs_keys[i]], irange)
                x, y
            end
        end
    else
        xlabel --> "Monte Carlo Time"
        ylabel --> "$(observable)"
        palette --> default_colors

        for name in obs_keys
            @series begin
                label --> name
                y = view(getproperty(m, observable)[name], irange)
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
RecipesBase.@recipe function biaspotential(
    bp::BiasPotential; cvlims=nothing, normalize=false, ylims=nothing
)
    b = bp.args[1]
    bias = b.bias
    xlims = cvlims ≡ nothing ? bias.cvlims : cvlims
    yylims = ylims ≡ nothing ? :auto : ylims
    isinf(sum(xlims)) && (xlims = (-6, 6))

    legend := false
    xticks --> floor(xlims[1]):ceil(xlims[2])
    xlims := (xlims[1], xlims[2])
    ylims := yylims
    xlabel --> "Collective Variable"
    ylabel --> "Bias Potential ($(typeof(bias)))"
    title --> b.ensemblename
    titlefontsize --> 10
    x = (bias isa OPES) ? (xlims[1]:0.001:xlims[2]-0.001) : bias.bin_vals
    yraw = b.(x)
    y = normalize ? yraw .- maximum(yraw) : yraw
    return x, y
end

@userplot HadronCorrelator

"""
    hadroncorrelator(m::MetaMeasurements, corr::Symbol; logscale=false, style=:line, tf=0)

Plot the effective mass plot of the hadron correlator `correlator` from the measurements in `m`.
"""
RecipesBase.@recipe function hadroncorrelator(
    hc::HadronCorrelator; logscale=false, style=:line, tf=0
)
    size --> (600, 500)
    link := :x
    layout := (2, 1)
    m, correlator = hc.args[1:2]
    @assert correlator ∈ observables(m) "Observable $correlator is not in Measurements"
    seriestype := style
    obs_keys = collect(keys(getproperty(m, correlator)))
    filter!(x -> x ≠ "itrj", obs_keys)
    filter!(x -> x ≠ "C" && x ≠ "C_flowed", obs_keys)
    palette --> default_colors
    x = collect(1:length(obs_keys))
    len = length(x)
    C = zeros(len)
    Cr = zeros(len)
    meff = zeros(len)
    tmp = last.(split.(obs_keys, "_"))

    if tf > 0
        tmp = split.(tmp, " ")
        tmp = [tmp[i][1] for i in eachindex(tmp)]
    end

    nums = parse.(Int, tmp)
    corr = first(split(string(correlator), "_"))
    str(it) = tf > 0 ? "$(corr)_corr_$(it) (tf=$tf)" : "$(corr)_corr_$(it)"

    for it in nums
        tmp = getproperty(m, correlator)[str(it)]
        C[it] = sum(tmp) / length(tmp)
    end

    key_str = tf > 0 ? "C_flowed" : "C"
    haskey(getproperty(m, correlator), key_str) || (getproperty(m, correlator)[key_str] = C)

    for it in nums
        Cr[it] = log(C[it] / C[mod1(it + 1, len)])
        meff[it] = try
            acosh((C[mod1(it + 1, len)] + C[mod1(it - 1, len)]) / 2C[it])
        catch _ 
            0.0
        end
    end

    xlabel --> "Time Extent"
    linecolor := default_colors[1]
    markercolor := default_colors[1]
    markershape := :circ

    @series begin
        subplot := 1
        xticks := 1:len
        ylabel --> "⟨C(t)⟩"
        label --> string(correlator)
        yscale := logscale ? :log10 : :identity
        y = C
        x, y
    end

    @series begin
        subplot := 2
        xticks := 1:len
        ylabel --> "m_eff"
        label --> string(correlator)
        yscale --> :identity
        y = meff
        x, y
    end
end

@userplot Eigenvalues

"""
    eigenvalues(m::MetaMeasurements)

Plot the `nev` mean eigenvalues in `m`.
"""
RecipesBase.@recipe function eigenvalues(ev::Eigenvalues; tf=0, xlims=(-1, 16), ylims=(-6, 6))
    m = ev.args[1]
    obs_sym = tf > 0 ? :eigenvalues_flowed : :eigenvalues
    @assert obs_sym ∈ observables(m) "Observable tmp is not in Measurements"
    obs_keys = collect(keys(getproperty(m, obs_sym)))
    filter!(x -> x ≠ "itrj", obs_keys)
    tf > 0 && filter(x -> x ∉ ("iflow", "tflow"), obs_keys)
    tmp = last.(split.(obs_keys, "_"))

    if tf > 0
        tmp = split.(tmp, " ")
        tmp = [tmp[i][1] for i in eachindex(tmp)]
    end

    nums = unique(parse.(Int, tmp))
    yre = zeros(length(nums))
    yim = zeros(length(nums))
    str(i, t) = tf > 0 ? "eig_$(t)_$(i) (tf=$(tf))" : "eig_$(t)_$(i)"
    
    for i in unique(nums)
        tmpre = getproperty(m, obs_sym)[str(i, "re")]
        tmpim = getproperty(m, obs_sym)[str(i, "im")]
        # yre[i] = sum(tmpre) / length(tmpre)
        # yim[i] = sum(tmpim) / length(tmpim)
        yre[i] = tmpre[end]
        yim[i] = tmpim[end]
    end

    seriestype := :scatter
    xlabel --> "Re(λ)"
    ylabel --> "Im(λ)"
    xlims --> xlims
    ylims --> ylims
    markercolor := default_colors[1]
    markershape := :circ

    @series begin
        label --> "dirac eigenvalues"
        yre, yim
    end
end

end
