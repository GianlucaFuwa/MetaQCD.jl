function fermion_parameters_from_dict(value_i::Dict)
    ferm_params = initialize_fermion_parameters()
    ferm_dict = struct2dict(ferm_params)

    for (key_ii, value_ii) in value_i
        if haskey(ferm_dict, key_ii)
            if !isnothing(value_ii)
                if key_ii == "mass"
                    setfield!(ferm_params, Symbol(key_ii), value_ii)
                else
                    keytype = typeof(getfield(ferm_params, Symbol(key_ii)))
                    setfield!(ferm_params, Symbol(key_ii), keytype(value_ii))
                end
            end
        end
    end

    value_out = deepcopy(ferm_params)
    return value_out
end

function initialize_fermion_parameters()
    return FermionParameters()
end

@kwdef mutable struct FermionParameters
    Nf::Int64 = 0
    mass::Union{Float64,Vector{Float64}} = [0.0]
    precon::String = "none"
    cg_tol_action::Float64 = 1e-10
    cg_tol_md::Float64 = 1e-7
    cg_maxiters_action::Int64 = 5000
    cg_maxiters_md::Int64 = 5000
    rhmc_spectral_bound::NTuple{2,Float64} = (1e-6, 64.0)
    rhmc_recalc_spectral_bound::Bool = false
    rhmc_order_action::Int64 = 15
    rhmc_order_md::Int64 = 15
    rhmc_prec_action::Int64 = 64
    rhmc_prec_md::Int64 = 64
    rhmc_tol_action::Float64 = 1e-7
    rhmc_tol_md::Float64 = 1e-6
end
