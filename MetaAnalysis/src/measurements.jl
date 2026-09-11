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
    tau_int::Dict{String,NTuple{2,Float64}}
    ensemblename::String
    function MetaMeasurements(ensemblename::String; full=false)
        _dir = if isabspath(ensemblename)
            @assert isdir(ensemblename) """
            Ensemble \"$(ensemblename)\" could not be found or doesn't exist.
            """
            ensemblename
        else
            path = joinpath(splitpath(@__DIR__())[1:end-2]...) * "/ensembles/$(ensemblename)/"
            @assert isdir(path) """
            Ensemble \"$(ensemblename)\" could not be found or doesn't exist.
            """
            path
        end

        dir = full ? _dir : joinpath(_dir, "measurements")
        @assert isdir(dir) "Directory $(dir) does not exist"
        hmc_logfile = "$(dir)/logs/hmc_acc_logs.txt"
        measurement_dict = Dict{String,Dict{String,Vector{Float64}}}()

        filenames = readdir(dir)
        isfile(hmc_logfile) && push!(filenames, hmc_logfile)
        tau_int = Dict{String,NTuple{2,Float64}}()

        for name in filenames
            name_no_ext = splitext(name)[1]
            instance = name_no_ext[end-2:end]
            measurement = Dict{String,Vector{Float64}}()
            if contains(name, "hmc_acc_logs")
                data, header = readdlm(hmc_logfile; header=true)
                
                for i in eachindex(header)
                    measurement[header[i]] = data[:, i]
                end

                measurement_dict["hmc_data"] = measurement
            elseif any(occursin.(("flowed", "gflow", "cooling"), name_no_ext))
                data, header = readdlm(dir * "/$(name)"; header=true)
                unique_tflow = unique(data[:, 3])
                unique_indices = Vector{Int64}[]
                measurement["itrj"] = data[1:length(unique_tflow):end, 1]
                
                for tflow in unique_tflow
                    push!(unique_indices, findall(x -> isapprox(tflow, x), data[:, 3]))
                end

                for i in 4:length(header)
                    head = header[i]
                    for (j, tflow) in enumerate(unique_tflow)
                        ui = unique_indices[j]
                        str = " (tf=$(tflow))"
                        measurement[head * str] = data[ui, i]

                        if header[i] != "itrj"
                            if header[i] == "Q_clover"
                                tau_int[head * str] = autoc_time_int(data[ui, i])
                                tau_int[head * "^2" * str] = autoc_time_int(data[ui, i].^2)
                            else
                                tau_int[head * str] = autoc_time_int(data[ui, i])
                            end
                        end
                    end
                end

                measurement_dict[name_no_ext] = measurement
            else
                data, header = readdlm(dir * "/$(name)"; header=true)

                for i in eachindex(header)
                    measurement[header[i]] = data[:, i]

                    if header[i] != "itrj"
                        tau_int[header[i] * "_$(instance)"] = autoc_time_int(data[:, i])
                    end
                end

                measurement_dict[name_no_ext] = measurement
            end
        end

        obs_sym = Symbol.(keys(measurement_dict))
        clear_wspace!()
        return new(measurement_dict, obs_sym, tau_int, ensemblename)
    end
end

Base.length(m::MetaMeasurements, observable) = Int(getproperty(m, observable)["itrj"][end])


function Base.getproperty(m::MetaMeasurements, s::Symbol)
    s == :ensemblename && return getfield(m, :ensemblename)
    s == :measurement_dict && return getfield(m, :measurement_dict)
    s == :observables && return getfield(m, :observables)
    s == :tau_int && return getfield(m, :tau_int)
    valid_names = Symbol.(keys(m.measurement_dict))
    @assert s ∈ valid_names "Your MetaMeasurements don't contain the observable $s"
    return getfield(m, :measurement_dict)["$(s)"]
end

observables(m::MetaMeasurements) = m.observables
auto_correlation(m::MetaMeasurements) = m.tau_int

function Base.show(io::IO, m::MetaMeasurements)
    print(io, "MetaMeasurements(ensemble: \"$(m.ensemblename)\")")
    return nothing
end

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
    if observable != :bias_data
        @assert observable ∈ observables(m) "Observable $observable is not in Measurements"
    end
    seriestype := seriestype
    obs_keys = if observable != :bias_data
        collect(keys(getproperty(m, observable)))
    else
        numinstances = 0
        while true
            if Symbol(:bias_data_, Symbol(lpad(numinstances, 3, "0"))) ∈ observables(m)
                numinstances += 1
            else
                break
            end
        end
        @show numinstances
        [collect(keys(getproperty(m, Symbol(:bias_data_, Symbol(lpad(i, 3, "0")))))) for i in 0:numinstances-1]
    end
    
    if observable != :bias_data
        filter!(x -> x ≠ "itrj", obs_keys)
    else
        [filter!(x -> x ≠ "itrj", obs_keys_i) for obs_keys_i in obs_keys]
    end

    palette --> DEFAULT_COLORS

    x = try
        view(getproperty(m, observable)["itrj"], irange)
    catch _
        nothing
    end

    if observable == :bias_data
        [filter!(x -> !contains("cv", x), obs_keys_i) for obs_keys_i in obs_keys]
        size --> (600, 400)
        link := :x
        legend := false
        palette --> DEFAULT_COLORS
        xlabel --> "Monte Carlo Time"
        ylabel --> "cv"

        for (i, name) in enumerate(obs_keys)
            @series begin
                color --> DEFAULT_COLORS[mod1(i+1, length(DEFAULT_COLORS))]
                y = view(getproperty(m, Symbol(:bias_data_, Symbol(lpad(i-1, 3, "0"))))["cv1"], irange)
                x, y
            end
        end
    elseif occursin("bias_data", string(observable))
        filter!(x -> !contains("cv", x), obs_keys)
        size --> (600, 200 * length(obs_keys))
        link := :x
        layout := (length(obs_keys), 1)
        legend := false
        palette --> DEFAULT_COLORS

        for (i, name) in enumerate(obs_keys)
            @series begin
                xl = i == length(obs_keys) ? "Monte Carlo Time" : ""
                xlabel --> xl
                ylabel --> name
                color --> DEFAULT_COLORS[i+1]
                subplot := i
                y = view(getproperty(m, observable)[name], irange)
                x, y
            end
        end
    elseif occursin("hmc_data", string(observable))
        palette --> DEFAULT_COLORS
        xlabel --> "Monte Carlo Time"
        ylabel --> "$(observable)"

        # for name in obs_keys
        @series begin
            label --> "ΔH"
            y = view(getproperty(m, observable)["ΔH"], irange)
            x, y
        end
        # end
    elseif any(occursin.(("flowed", "gflow", "cooling"), string(observable)))
        palette --> DEFAULT_COLORS
        xlabel --> "Monte Carlo Time"
        linewidth --> 2
        legend --> :outertopright

        sub_obs = unique!(first.(split.(obs_keys, " ")))
        size --> (600, 250 * length(sub_obs))
        layout := (length(sub_obs), 1)
        nlabel = last.(split.(obs_keys, " "))
        tf_digits = parse.(Float64, filter.(x -> isdigit(x) || x=='.', nlabel))

        for tflow in sort(unique(tf_digits))
            (isnothing(tf) || tflow==tf) || continue
            for (j, sub_ob) in enumerate(sub_obs)
                @series begin
                    # subplot := j
                    ylabel --> sub_ob
                    label --> "tf = $(tflow)"
                    y = view(getproperty(m, observable)["$(sub_ob) (tf=$(tflow))"], irange)
                    x, y
                end
            end
        end
    else
        xlabel --> "Monte Carlo Time"
        ylabel --> "$(observable)"
        palette --> DEFAULT_COLORS

        for name in obs_keys
            @series begin
                label --> name
                y = view(getproperty(m, observable)[name], irange)
                x, y
            end
        end
    end
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
    if any(occursin.(("flowed", "gflow", "cooling"), string(correlator)))
        @assert tf != 0 "specific flow time 'tf' needs to be given as keyword argument with flowed correlator"
    else
        tf = 0
    end
    seriestype := style
    obs_keys = collect(keys(getproperty(m, correlator)))
    filter!(x -> x ≠ "itrj", obs_keys)
    filter!(x -> x ≠ "C" && x ≠ "C_flowed", obs_keys)
    palette --> DEFAULT_COLORS
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
    linecolor := DEFAULT_COLORS[1]
    markercolor := DEFAULT_COLORS[1]
    markershape := :circ

    @series begin
        subplot := 1
        xticks := 1:len
        ylabel --> "⟨C(t)⟩ (tf = $tf)"
        label --> string(correlator)
        yscale := logscale ? :log10 : :identity
        y = C
        x, y
    end

    @series begin
        subplot := 2
        xticks := 1:len
        ylabel --> "m_eff (tf = $tf)"
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
    markercolor := DEFAULT_COLORS[1]
    markershape := :circ

    @series begin
        label --> "dirac eigenvalues"
        yre, yim
    end
end
