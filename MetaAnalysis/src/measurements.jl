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
    ensemblepath::String
    function MetaMeasurements(ensemblename::String; name=nothing)
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

        dir = joinpath(_dir, "measurements")
        logdir = joinpath(_dir, "logs")
        @assert isdir(dir) "Directory $(dir) does not exist"
        measurement_dict = Dict{String,Dict{String,Vector{Float64}}}()

        filenames = readdir(dir)
        logfilenames = isdir(logdir) ? readdir(logdir) : String[]
        filter!(x -> contains(x, "hmc_acc_logs"), logfilenames)
        append!(filenames, logfilenames)
        tau_int = Dict{String,NTuple{2,Float64}}()

        for name in filenames
            occursin("cg_data", name) && continue
            name_no_ext = splitext(name)[1]
            instance = name_no_ext[end-2:end]
            measurement = Dict{String,Vector{Float64}}()
            if contains(name, "hmc_acc_logs")
                header = ["ΔP2", "ΔSg", "ΔSf", "ΔV", "ΔH", "Total Action", "Accepted"]
                data = readdlm(logdir * "/$(name)"; skipstart=1)
                
                for i in eachindex(header)
                    measurement[header[i]] = data[10:end, i]
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
                        str = "_$(instance) (tf=$(tflow))"
                        measurement[head * str] = data[ui, i]

                        if header[i] != "itrj"
                            if header[i] == "Q_clover"
                                tau_int[head * str] = try
                                    autoc_time_int_uw(data[ui, i])
                                catch _
                                    @warn "Autocorrelation time of $(head * str) could not be determined using UWerr, falling back to manual"
                                    autoc_time_int(data[ui, i]), 0.0
                                end
                                tau_int[head * "^2" * str] = try
                                    autoc_time_int_uw(data[ui, i].^2)
                                catch _
                                    @warn "Autocorrelation time of $(head * "^2" * str) could not be determined using UWerr, falling back to manual"
                                    autoc_time_int(data[ui, i].^2), 0.0
                                end
                            else
                                tau_int[head * str] = try
                                    autoc_time_int_uw(data[ui, i])
                                catch _
                                    @warn "Autocorrelation time of $(head * str) could not be determined using UWerr, falling back to manual"
                                    autoc_time_int(data[ui, i]), 0.0
                                end
                            end
                        end
                    end
                end

                measurement_dict[name_no_ext] = measurement
            else
                data, header = readdlm(dir * "/$(name)"; header=true)

                for i in 2:length(header)
                    measurement[header[i]] = data[:, i]

                    if header[i] != "itrj"
                        tau_int[header[i] * "_$(instance)"] = try 
                            autoc_time_int_uw(data[:, i])
                        catch _
                            @warn "Autocorrelation time of $(header[i] * "_$(instance)") could not be determined using UWerr, falling back to manual"
                            autoc_time_int(data[:, i]), 0.0
                        end
                    end
                end

                measurement_dict[name_no_ext] = measurement
            end
        end

        obs_sym = Symbol.(keys(measurement_dict))
        _name = isnothing(name) ? ensemblename : name
        clear_wspace!()
        return new(measurement_dict, obs_sym, tau_int, _name, _dir)
    end
end

Base.length(m::MetaMeasurements, observable) = Int(getproperty(m, observable)["itrj"][end])


function Base.getproperty(m::MetaMeasurements, s::Symbol)
    s == :ensemblename && return getfield(m, :ensemblename)
    s == :ensemblepath && return getfield(m, :ensemblepath)
    s == :measurement_dict && return getfield(m, :measurement_dict)
    s == :observables && return getfield(m, :observables)
    s == :tau_int && return getfield(m, :tau_int)
    valid_names = Symbol.(keys(m.measurement_dict))
    @assert s ∈ valid_names "Your MetaMeasurements don't contain the observable $s"
    return getfield(m, :measurement_dict)["$(s)"]
end

observables(m::MetaMeasurements) = m.observables
auto_correlation(m::MetaMeasurements) = m.tau_int

function Base.show(io::IO, ::MIME"text/plain", m::MetaMeasurements)
    print(io, "MetaMeasurements(ensemble: \"$(m.ensemblename)\")")
    return nothing
end

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
        for obs_keys_i in obs_keys
            filter!(x -> contains(x, "cv"), obs_keys_i)
        end

        numplots = length(obs_keys[1])
        size --> (600, 300*numplots)
        link := :x
        legend := false
        layout := (numplots, 1)
        palette --> DEFAULT_COLORS

        for (i, name) in enumerate(obs_keys)
            for j in 1:numplots
                @series begin
                    xlab = j == numplots ? "Monte Carlo Time" : ""
                    color --> DEFAULT_COLORS[mod1(i+1, length(DEFAULT_COLORS))]
                    ylabel --> "cv$(j)"
                    xlabel --> xlab
                    y = view(getproperty(m, Symbol(:bias_data_, Symbol(lpad(i-1, 3, "0"))))["cv$(j)"], irange)
                    subplot := j
                    x, y
                end
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
        legend --> :outertopright

        sub_obs = unique!(first.(split.(obs_keys, " ")))
        size --> (600, 300 * length(sub_obs))
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
    hc::HadronCorrelator; logscale=false, style=:line, tf=0, with_errs=false,
    fit_plateau=false, plateau_range=nothing, staggered=false
)
    fit_plateau && !with_errs && throw(ArgumentError(
        "fit_plateau=true requires with_errs=true so the plateau error can be obtained via ADerrors.jl"
    ))

    sessid = rand(UInt64)
    size --> (600, 500)
    link := :x
    layout := (2, 1)
    m, correlator = hc.args[1:2]
    savedir = joinpath(m.ensemblepath, "analysis")
    corrname = "$(join(split(string(correlator), "_")[1:end-2], " ")) (tf=$tf)"
    is_gflow = contains(String(correlator), "gflow")
    @assert correlator ∈ observables(m) "Observable $correlator is not in Measurements"
    tf > 0 && @assert is_gflow
    is_gflow && @assert tf > 0
    seriestype := style
    obs_keys = collect(keys(getproperty(m, correlator)))
    filter!(x -> x ≠ "itrj", obs_keys)
    filter!(x -> x ≠ "C" && x ≠ "C_flowed", obs_keys)
    filter!(x -> contains(x, "tf=$(tf)"), obs_keys)
    palette --> DEFAULT_COLORS
    x = 1:length(obs_keys)
    T = length(x)
    corrrange = 2:div(T, 2)+1
    @show T, corrrange[end]
    T2 = length(corrrange)
    C = []
    Cr = []
    meff = []
    tmp = if is_gflow
        [split(obs_keys[i], "_")[3] for i in eachindex(obs_keys)]
    else
        last.(split.(obs_keys, "_"))
    end
    nums = sort(parse.(Int, tmp))
    corr = first(split(string(correlator), "_"))
    str(it) = is_gflow ? "$(corr)_corr_$(it)_000 (tf=$tf)" : "$(corr)_corr_$(it)"
    for it in corrrange
        _tmp1 = getproperty(m, correlator)[str(it)]
        _tmp2 = getproperty(m, correlator)[str(T-it+2)]
        tmp2 = if with_errs
            try
                if it != div(T, 2)+1
                    u1 = uwreal(_tmp1, "$(str(it)) $(sessid)")
                    u2 = uwreal(_tmp2, "$(str(it)) $(sessid)")
                    u = (u1 + u2) / 2
                else
                    u = uwreal(_tmp1, "$(str(it)) $(sessid)")
                end
                uwerr(u)
                u
            catch _
                if it != div(T, 2)+1
                    r1 = analyze(_tmp1, Bootstrap())
                    r2 = analyze(_tmp2, Bootstrap())
                    u1 = uwreal([r1["mean"], r1["stderr"]], "$(str(it)) $(sessid) bb")
                    u2 = uwreal([r2["mean"], r2["stderr"]], "$(str(it)) $(sessid) bb")
                    u = (u1 + u2) / 2
                else
                    r = analyze(_tmp1, Bootstrap())
                    u = uwreal([r["mean"], r["stderr"]], "$(str(it)) $(sessid) bb")
                end
                uwerr(u)
                u
            end
        else
            (sum(_tmp1)/length(_tmp1) + sum(_tmp2)/length(_tmp2)) / 2
        end
        push!(C, tmp2)
    end
    for it in 2:corrrange[end]-2
        # tmp = log(C[it] / C[mod1(it + 1, T)])
        tmp2 = try
            if it != corrrange[end]-1
                if staggered
                    # 0.5acosh((C[mod1(it + 2, T2)] + C[mod1(it - 2, T2)]) / 2C[it])
                    if C[mod1(it + 1, T2)] == C[end]
                        0.5 * (
                            acosh(C[mod1(it - 1, T2)] / C[end])
                        )
                    else
                        0.5 * (
                            acosh(C[mod1(it - 1, T2)] / C[end]) -
                            acosh(C[mod1(it + 1, T2)] / C[end])
                        )
                    end
                else
                    acosh((C[mod1(it + 1, T2)] + C[mod1(it - 1, T2)]) / 2C[it])
                end
            else
                nothing
            end
        catch _
            with_errs ? uwreal([0.0, 0.0], it) : 0.0
        end
        if with_errs
            # uwerr(tmp)
            !isnothing(tmp2) && uwerr(tmp2)
        end
        # push!(Cr, tmp)
        !isnothing(tmp2) && push!(meff, tmp2)
    end
    if with_errs
        # uwerr.(Cr)
        uwerr.(meff)
    end

    # Write correlator means and meff to file
    mkpath(savedir)
    io = open(joinpath(savedir, corrname), "w")
    println(io, "$(rpad("t", 5, " "))$(rpad("mean", 25, " "))$(rpad("err", 25, " "))")
    for i in eachindex(C)
        @printf io "%-5i" i
        @printf io "%-25.15E" value(C[i])
        @printf io "%-25.15E\n" ADerrors.err(C[i])
    end
    close(io)

    # --- plateau fit -------------------------------------------------
    meff_plateau = nothing
    pval = perr = nothing
    prange = 1:0
    if fit_plateau
        prange = something(plateau_range, (corrrange[end]-3-T2÷3:corrrange[end]-3))
        prange = collect(prange)
        @assert prange ⊆ corrrange "plateau_range $(prange) is out of bounds $(corrrange)"

        # drop any points whose error collapsed to zero (failed acosh fallback)
        selected = filter(i -> ADerrors.err(meff[i]) > 0, prange)
        isempty(selected) && error(
            "No valid points in plateau_range=$(prange) to fit (all have zero error)."
        )

        w = 1 ./ ADerrors.err.(meff[selected]) .^ 2
        meff_plateau = sum(w .* meff[selected]) / sum(w)   # linear combo of uwreal ⇒ exact error propagation
        uwerr(meff_plateau)
        pval = value(meff_plateau)
        perr = ADerrors.err(meff_plateau)

        println(
            "Plateau fit [$(first(selected)), $(last(selected))] for $(corrname): " *
            "am_eff = $(pval) ± $(perr)"
        )
    end
    # -------------------------------------------------------------------

    io = open(joinpath(savedir, corrname*"_meff"), "w")
    if fit_plateau
        println(io, "# Plateau fit in range [$(first(selected)+1), $(last(selected)+1)]: am_eff = $(pval) ± $(perr)")
    end
    println(io, "$(rpad("t", 5, " "))$(rpad("mean", 25, " "))$(rpad("err", 25, " "))")
    for (i, m) in enumerate(meff)
        @printf io "%-5i" corrrange[i]
        @printf io "%-25.15E" value(m)
        @printf io "%-25.15E\n" ADerrors.err(m)
    end
    close(io)

    xlabel --> "Time Extent"
    linecolor := DEFAULT_COLORS[1]
    markercolor := DEFAULT_COLORS[1]
    markershape := :circ
    @series begin
        subplot := 1
        ylabel --> L"\langle C(t) \rangle"
        titlefontsize --> 10
        title --> "Ensemble: $(split(m.ensemblename, "/")[end])"
        label --> corrname
        ylims --> (minimum(Float64.(C))*0.5, maximum(Float64.(C))*1.3)
        yscale := logscale ? :log10 : :identity
        y = with_errs ? value.(C) : Float64.(C)
        if with_errs
            yerror := ADerrors.err.(C)
        end
        collect(corrrange).-1, y
    end
    @series begin
        subplot := 2
        ylabel --> L"am_\mathrm{eff}"
        label --> corrname
        ylims --> (-0.1, maximum(Float64.(meff))*1.1)
        yscale --> :identity
        legend --> :topright
        y = with_errs ? value.(meff) : Float64.(meff)
        if with_errs
            yerror := ADerrors.err.(meff)
        end
        2:corrrange[end]-2, y
    end
    if fit_plateau
        @series begin
            subplot := 2
            seriestype := :path
            markershape := :none
            linecolor := DEFAULT_COLORS[2]
            fillalpha --> 0.25
            legend --> :topright
            fillcolor := DEFAULT_COLORS[2]
            ribbon := perr
            label --> "plateau: $(round(pval, digits=5)) ± $(round(perr, digits=5))"
            corrrange[prange], fill(pval, length(prange))
        end
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
