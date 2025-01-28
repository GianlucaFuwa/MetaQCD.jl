abstract type BiasParameters end

function struct2dict(x::T) where {T}
    return Dict{String,Any}(string(fn) => getfield(x, fn) for fn in fieldnames(T))
end

function bias_parameters_from_dict(value_i::Dict)
    kind_of_bias = value_i["kind_of_bias"]
    bias_params = initialize_bias_parameters(kind_of_bias)
    bias_dict = struct2dict(bias_params)

    for (key_ii, value_ii) in value_i
        key_ii == "kind_of_bias" && continue

        if haskey(bias_dict, key_ii)
            if !isnothing(typeof(value_ii))
                keytype = typeof(getfield(bias_params, Symbol(key_ii)))
                setfield!(bias_params, Symbol(key_ii), keytype(value_ii))
            end
        end
    end

    value_out = deepcopy(bias_params)
    return value_out
end

function initialize_bias_parameters(kind_of_bias)
    if Unicode.normalize(kind_of_bias; casefold=true) ∈ ("metad", "metadynamics")
        method = MetadynamicsParameters()
    elseif Unicode.normalize(kind_of_bias; casefold=true) == "opes"
        method = OPESParameters()
    elseif Unicode.normalize(kind_of_bias; casefold=true) == "opesmt"
        method = OPESmultithermalParameters()
    elseif Unicode.normalize(kind_of_bias; casefold=true) == "parametric"
        method = ParametricParameters()
    else
        error("$(kind_of_bias) is not implemented")
    end

    return method
end

Base.@kwdef mutable struct MetadynamicsParameters <: BiasParameters
    name::String = "metadynamics"
    kind_of_cv::String = "topcharge_clover"
    usebiases::Vector{String} = String[]
    static::Bool = false
    numsmears_for_cv::Int64 = 4
    symmetric::Bool = false
    stride::Int64 = 1
    write_bias_every::Int64 = stride
    cvlims::Vector{Float64} = [-3.0, 3.0]
    biasfactor::Float64 = Inf
    bin_width::Float64 = 0.02
    weight::Float64 = 0.02
    penalty_weight::Float64 = 100
end

Base.@kwdef mutable struct OPESParameters <: BiasParameters
    name::String = "opes"
    kind_of_cv::String = "topcharge_clover"
    usebiases::Vector{String} = String[]
    static::Bool = false
    numsmears_for_cv::Int64 = 4
    stride::Int64 = 1
    write_bias_every::Int64 = stride
    symmetric::Bool = false
    explore::Bool = false
    cvlims::Vector{Float64} = [-3.0, 3.0]
    barrier::Float64 = 10.0
    biasfactor::Float64 = Inf
    sigma_0::Float64 = 0.02
    sigma_min::Float64 = 1e-4
    fixed_sigma::Bool = false
    adaptive_Z::Bool = false 
    epsilon::Float64 = 0.0 
    threshold::Float64 = 1.0
    cutoff::Float64 = 0.0
    penalty_weight::Float64 = 100
end

Base.@kwdef mutable struct OPESmultithermalParameters <: BiasParameters
    name::String = "opesmt"
    kind_of_cv::String = "multithermal"
    usebiases::Vector{String} = String[]
    numsmears_for_cv::Int64 = 0
    static::Bool = false
    stride::Int64 = 1
    write_bias_every::Int64 = stride
    beta_min_max::Vector{Float64} = []
    beta_num::Int64 = 2
end

Base.@kwdef mutable struct ParametricParameters <: BiasParameters
    name::String = "parametric"
    kind_of_cv::String = "topcharge_clover"
    usebiases::Vector{String} = String[]
    static::Bool = true
    numsmears_for_cv::Int64 = 4
    cvlims::Vector{Float64} = [-3.0, 3.0]
    penalty_weight::Float64 = 100
    Q::Float64 = 0.0
    A::Float64 = 0.0
    Z::Float64 = 0.0
end

function get_cvinfo_from_parameters(p::BiasParameters)
    cv_func = if p.kind_of_cv == "topcharge_plaquette"
        U -> top_charge(Plaquette(), U)
    elseif p.kind_of_cv == "topcharge_clover"
        U -> top_charge(Clover(), U)
    elseif p.kind_of_cv == "multithermal"
        @assert p.name == "opesmt" "Multithermal CV only works with opesmt"
        U -> calc_gauge_action(U)
    else
        error("kind_of_cv \"$(p.kind_of_cv)\" not supported (see docs for supported CVs)")
    end

    deriv_func = if p.kind_of_cv == "topcharge_plaquette"
        (dU, F, U, fac) -> top_charge_deriv!(dU, F, U, Plaquette(), fac)
    elseif p.kind_of_cv == "topcharge_clover"
        (dU, F, U, fac) -> top_charge_deriv!(dU, F, U, Clover(), fac)
    elseif p.kind_of_cv == "multithermal"
        (dU, staples, U, fac) -> gauge_action_deriv!(dU, staples, U, fac)
    else
        error("kind_of_cv \"$(p.kind_of_cv)\" not supported (see docs for supported CVs)")
    end

    cv_temp_ind = if p.kind_of_cv in ("topcharge_plaquette", "topcharge_clover")
        Val(1)
    elseif p.kind_of_cv == "multithermal"
        Val(2)
    else
        error("kind_of_cv \"$(p.kind_of_cv)\" not supported (see docs for supported CVs)")
    end

    return CVinfo(cv_func, deriv_func, cv_temp_ind)
end
