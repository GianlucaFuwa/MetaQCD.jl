abstract type BiasParameters end

function bias_parameters_from_dict(input::Dict, instance=MPI_INSTANCE[]; build=false)
    type = input["type"]
    bias_params = initialize_bias_parameters(type)
    bias_dict = struct2dict(bias_params)

    for (key_i, value_i) in input
        key_i == "type" && continue

        if haskey(bias_dict, key_i)
            if !isnothing(value_i)
                if key_i == "static"
                    idx = if build
                        1
                    else
                        if length(value_i) == 1
                            1
                        else
                            @assert length(value_i) >= instance+1 """
                            if tempering is enabled, the 'static' parameter has to be a vector of length >= numinstances
                            """
                            instance+1
                        end
                    end

                    setfield!(bias_params, :static, Bool(value_i[idx]))
                elseif key_i == "load_bias"
                    setfield!(bias_params, :load_bias, String[value_i...])
                else
                    keytype = typeof(getfield(bias_params, Symbol(key_i)))
                    setfield!(bias_params, Symbol(key_i), keytype(value_i))
                end
            end
        end
    end

    out = deepcopy(bias_params)
    return out
end

function initialize_bias_parameters(type)
    if lowercase(type) ∈ ("metad", "metadynamics")
        method = MetadynamicsParameters()
    elseif lowercase(type) == "opes"
        method = OPESParameters()
    elseif lowercase(type) == "opesmt"
        method = OPESmultithermalParameters()
    elseif lowercase(type) == "ves"
        method = VESParameters()
    else
        error("$(type) is not implemented")
    end

    return method
end

@kwdef mutable struct MetadynamicsParameters <: BiasParameters
    type::String = "metadynamics"
    kind_of_cv::String = "topcharge_clover"
    load_bias::Vector{String} = String[]
    static::Bool = true
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

@kwdef mutable struct OPESParameters <: BiasParameters
    type::String = "opes"
    kind_of_cv::String = "topcharge_clover"
    load_bias::Vector{String} = String[]
    static::Bool = true
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

@kwdef mutable struct OPESmultithermalParameters <: BiasParameters
    type::String = "opesmt"
    kind_of_cv::String = "multithermal"
    load_bias::Vector{String} = String[]
    static::Bool = true
    numsmears_for_cv::Int64 = 0
    stride::Int64 = 1
    write_bias_every::Int64 = stride
    beta_min_max::Vector{Float64} = []
    beta_num::Int64 = 2
end

@kwdef mutable struct VESParameters <: BiasParameters
    type::String = "ves"
    kind_of_cv::String = "topcharge_clover"
    load_bias::Vector{String} = String[]
    static::Bool = true
    numsmears_for_cv::Int64 = 4
    cvlims::Vector{Float64} = [-3.0, 3.0]
    penalty_weight::Float64 = 100
    step_size::Float64 = 0.1
    nbasis::Int64 = 20
    batch_size::Int64 = 50
    alpha::Vector{Float64} = []
    write_bias_every::Int64 = batch_size
end

function get_cvinfo_from_parameters(p::BiasParameters)
    cv_func = if p.kind_of_cv == "topcharge_plaquette"
        U -> top_charge(Plaquette(), U)
    elseif p.kind_of_cv == "topcharge_clover"
        U -> top_charge(Clover(), U)
    elseif p.kind_of_cv == "multithermal"
        @assert p.type == "opesmt" "Multithermal CV only works with opesmt"
        U -> calc_gauge_action(U) / length(U)
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
