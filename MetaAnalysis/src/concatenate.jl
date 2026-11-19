const MINITRJ = Base.RefValue{Int64}(0)
const MAXITRJ = Base.RefValue{Int64}(0)

get_num(str) = parse(Int64, split(str, "_")[end])

function discover_observable_bases(ens_dirs::AbstractVector{<:AbstractString})
    bases = String[]

    for ens_dir in ens_dirs
        mdir = joinpath(ens_dir, "measurements")
        isdir(mdir) || continue
        for f in readdir(mdir)
            m = match(r"^(.+)_(\d{3})\.txt$", f)
            isnothing(m) && continue
            push!(bases, "$(m.captures[1])_$(m.captures[2])")
        end
    end

    return unique(bases)
end

function discover_log_bases(ens_dirs::AbstractVector{<:AbstractString})
    bases = String[]

    for ens_dir in ens_dirs
        ldir = joinpath(ens_dir, "logs")
        (isdir(ldir) || isempty(ldir)) || continue
        for f in readdir(ldir)
            l = match(r"^(.+)_(\d{3})\.txt$", f)
            (isnothing(l) || l.captures[1]=="logs") && continue
            push!(bases, "$(l.captures[1])_$(l.captures[2])")
        end
    end

    return unique(bases)
end

function load_raw_observable(ens_dir::String, base::String, checkpoint_interval)
    f = joinpath(ens_dir, "measurements", "$(base).txt")
    isfile(f) || return Float64[], zeros(0, 0), ""
    header = open(io -> readline(io), f)
    data = readdlm(f; skipstart=1)
    (isempty(data) || ndims(data) < 2) && return Float64[], zeros(0, 0), header
    maxitrj = fld(data[end, 1], checkpoint_interval) * checkpoint_interval
    MAXITRJ[] = maxitrj
    itrj = Int64[]
    rows = Vector{Vector{Float64}}()

    for i in 1:size(data, 1)
        data[i, 1] > maxitrj && break
        push!(itrj, Int64(data[i, 1]))
        push!(rows, Float64.(data[i, 2:end]))
    end

    values = isempty(rows) ? zeros(0, size(data, 2) - 1) : permutedims(hcat(rows...))
    return itrj, values, header
end

function load_raw_logs(ens_dir::String, base::String, checkpoint_interval, numtherm; first=false)
    f = joinpath(ens_dir, "logs", "$(base).txt")
    isfile(f) || return Float64[], zeros(0, 0), ""
    header = open(io -> readline(io), f)
    data = readdlm(f; skipstart=1)
    isempty(data) && return Float64[], zeros(0, 0), header
    offset = first ? numtherm : 0
    maxitrj = fld(MAXITRJ[], checkpoint_interval) * checkpoint_interval
    rows = Vector{Vector{Float64}}()
    start = first ? 1 : MINITRJ[]+1
    finish = first ? numtherm+maxitrj : maxitrj
    rng = start:finish
    evals_per_trj = fld(length(data[:, 1]), length(rng))

    for itrj in eachindex(rng)
        for i in 1:evals_per_trj
            push!(rows, Float64.(data[(itrj-1)*evals_per_trj+i, :]))
        end
    end

    values = isempty(rows) ? zeros(0, size(data, 2) - 1) : permutedims(hcat(rows...))
    return values, header
end

function match_runs(dir::String, pattern::String)
    all_names = readdir(dir; join=false)
    all_matched = all_names[findall(x -> contains(x, pattern), all_names)]
    out = String[]

    for imatched in eachindex(all_matched)
        push!(out, joinpath(dir, all_matched[imatched]))
    end

    return out
end

function sorted_ensemble_dirs(dirs::AbstractVector{<:AbstractString})
    return sort(dirs; lt=(x, y) -> isless(get_num(x), get_num(y)))
end

function concatenate_runs(
    runs::Vector{String},
    ensemble_name,
    checkpoint_interval,
    numtherm,
)
    @assert length(runs) > 0 "no runs with ensemble name $(ensemble_name) found"
    obs_bases   = discover_observable_bases(runs)
    filter!(x -> !contains(x, "cg_data"), obs_bases)
    obs_itrj    = Dict{String,Vector{Int64}}(b => Int64[] for b in obs_bases)
    obs_values  = Dict{String,Matrix{Float64}}()
    obs_headers = Dict{String,String}()
    log_bases   = discover_log_bases(runs)
    log_values  = Dict{String,Matrix{Float64}}()
    log_headers = Dict{String,String}()

    for (iens, ens_dir) in enumerate(sorted_ensemble_dirs(runs))
        @show ens_dir
        first = iens == 1
        # last = iens == length(runs)

        for base in obs_bases
            @show base
            oi, ov, oh = load_raw_observable(ens_dir, base, checkpoint_interval)
            isempty(oi) && continue
            append!(obs_itrj[base], oi)
            obs_values[base] = haskey(obs_values, base) ? vcat(obs_values[base], ov) : ov
            haskey(obs_headers, base) || (obs_headers[base] = oh)
        end

        for base in log_bases
            @show base, ens_dir
            lv, lh = load_raw_logs(ens_dir, base, checkpoint_interval, numtherm; first)
            isempty(lv) && continue
            log_values[base] = haskey(log_values, base) ? vcat(log_values[base], lv) : lv
            haskey(log_headers, base) || (log_headers[base] = lh)
        end

        MINITRJ[] = MAXITRJ[]
    end

    ensemble_dir = mktempdir()
    new_ensemblepath = joinpath(ensemble_dir, ensemble_name)
    mkpath(new_ensemblepath)
    mkpath(joinpath(new_ensemblepath, "measurements"))
    mkpath(joinpath(new_ensemblepath, "logs"))

    for base in obs_bases
        open(joinpath(new_ensemblepath, "measurements/$(base).txt"), "w+") do f
            write(f, obs_headers[base] * "\n")
            for i in eachindex(obs_itrj[base])
                @printf f "%-11i" obs_itrj[base][i]
                if contains(base, "gflow")
                    @printf f "%-7i" obs_values[base][i, 1]
                    @printf f "%-9.5f" obs_values[base][i, 2]
                    for iv in 3:size(obs_values[base], 2)
                        @printf f "%-25.15E" obs_values[base][i, iv]
                    end
                else
                    for iv in 1:size(obs_values[base], 2)
                        @printf f "%-25.15E" obs_values[base][i, iv]
                    end
                end
                println(f)
            end
            # writedlm(f, [obs_itrj[base] obs_values[base]])
        end
    end

    for base in log_bases
        open(joinpath(new_ensemblepath, "logs/$(base).txt"), "w+") do f
            write(f, log_headers[base] * "\n")
            if contains(base, "hmc_force") || contains(base, "hmc_acc")
                for i in 1:size(log_values[base], 1)
                    for iv in 1:size(log_values[base], 2)
                        @printf f "%-25.15E" log_values[base][i, iv]
                    end
                    println(f)
                end
            else
                writedlm(f, log_values[base])
            end
        end
    end

    return new_ensemblepath
end
