module Universe

using AMDGPU: functional as roc_functional
using CUDA: functional as cuda_functional
using Dates
using Unicode
using ..Utils
using ..Output

import ..DiracOperators: WilsonFermionAction, StaggeredFermionAction
import ..Gaugefields: Gaugefield, CPU, CUDABackend, ROCBackend, WilsonGaugeAction
import ..Gaugefields: IwasakiGaugeAction, DBW2GaugeAction, SymanzikTreeGaugeAction
import ..Gaugefields: initial_gauges
import ..BiasModule: Bias
import ..Parameters: ParameterSet

"""
    Univ(p::ParameterSet; use_mpi=false)

Create a Universe, containing the gauge configurations that are updated throughout a 
simulation and the bias potentials, if any. \\
`use_mpi` is soley an indicator, that sets `numinstances` to 1 even if stated otherwise
in the parameters.
"""
struct Univ{TG,TF,TB}
    U::TG
    fermion_actions::TF
    bias::TB
    numinstances::Int64
end

function Univ(p::ParameterSet; use_mpi=false)
    @level1("┌ Setting Universe...")
    NX, NY, NZ, NT = p.L
    β = p.beta
    initial = p.initial
    B = get_backend_from_string(p.backend)
    T = get_floatT_from_string(p.float_type)
    GA = get_gaugeaction_from_string(p.gauge_action)
    @level1("|  Backend: $(B)")
    @level1("|  Floating Point Precision: $(T)")
    @level1("|  L: $(NX)x$(NY)x$(NZ)x$(NT)")
    @level1("|  Gauge Action: $(GA)")
    @level1("|  beta: $β")

    if p.kind_of_bias != "none"
        if p.tempering_enabled && use_mpi == false
            numinstances = p.numinstances
            @level1("|  Using 1 + $(numinstances-1) instances\n")
            U₁ = initial_gauges(initial, NX, NY, NZ, NT, β; BACKEND=B, T=T, GA=GA)
            fermion_actions₁ = init_fermion_actions(p, U₁)
            bias₁ = Bias(p, U₁; instance=0) # instance=0 -> dummy bias for non-MetaD stream

            U = Vector{typeof(U₁)}(undef, numinstances)
            fermion_actions = Vector{typeof(fermion_actions₁)}(undef, numinstances)
            bias = Vector{typeof(bias₁)}(undef, numinstances)

            U[1] = U₁
            fermion_actions[1] = fermion_actions₁
            bias[1] = bias₁
            for i in 2:numinstances
                U[i] = initial_gauges(initial, NX, NY, NZ, NT, β; BACKEND=B, T=T, GA=GA)
                fermion_actions[i] = deepcopy(fermion_actions₁)
                bias[i] = Bias(p, U[i]; instance=i - 1)
            end
        else
            numinstances = 1
            U = initial_gauges(initial, NX, NY, NZ, NT, β; BACKEND=B, T=T, GA=GA)
            fermion_actions = init_fermion_actions(p, U)
            bias = Bias(p, U)
        end
    else
        @assert p.tempering_enabled == false "tempering can only be enabled with bias"
        numinstances = 1
        U = initial_gauges(initial, NX, NY, NZ, NT, β; BACKEND=B, T=T, GA=GA)
        fermion_actions = init_fermion_actions(p, U)
        bias = nothing
    end

    @level1("└\n")
    return Univ{typeof(U),typeof(fermion_actions),typeof(bias)}(
        U, fermion_actions, bias, numinstances
    )
end

function get_backend_from_string(backend::String)
    backend_str = Unicode.normalize(backend; casefold=true)

    if backend_str == "cpu"
        backend = CPU
    elseif backend_str ∈ ("cuda", "cudabackend")
        @assert cuda_functional(true) """
            Your machine either does not have a CUDA compatible GPU
            or the CUDA stack is not installed
            """
        backend = CUDABackend
    elseif backend_str ∈ ("roc", "rocbackend")
        @assert roc_functional() """
            Your machine either does not have a ROCm compatible GPU
            or the ROCm stack is not installed
            """
        backend = ROCBackend
    else
        error("Backend \"$(backend)\" not supported")
    end

    return backend
end

function get_floatT_from_string(float_type::String)
    float_type_str = Unicode.normalize(float_type; casefold=true)

    if float_type_str ∈ ("float64", "double")
        FloatT = Float64
    elseif float_type_str ∈ ("float32", "single")
        FloatT = Float32
    elseif float_type_str ∈ ("float16", "half")
        FloatT = Float16
    else
        error("Float64, Float32 or Float16 float types supported, not \"$(float_type)\"")
    end

    return FloatT
end

function get_gaugeaction_from_string(gaction::String)
    gauge_action = Unicode.normalize(gaction; casefold=true)

    if gauge_action == "wilson"
        GA = WilsonGaugeAction
    elseif gauge_action == "symanzik_tree"
        GA = SymanzikTreeGaugeAction
        # elseif gauge_action == "symanzik_tadpole"
        #     GA = SymanzikTadGaugeAction
    elseif gauge_action == "iwasaki"
        GA = IwasakiGaugeAction
    elseif gauge_action == "dbw2"
        GA = DBW2GaugeAction
    else
        error("Gauge action \"$(gaction)\" not supported")
    end

    return GA
end

function init_fermion_actions(p::ParameterSet, U)
    fermion_action = p.fermion_action

    if fermion_action == "none"
        fermion_actions = nothing
    elseif fermion_action == "wilson"
        Sf_light = WilsonFermionAction(
            U,
            p.mass_light;
            Nf=p.Nf_light,
            csw=p.csw,
            anti_periodic=p.anti_periodic,
            cg_tol=p.cg_tol,
            cg_maxiters=p.cg_maxiters,
        )
        if p.Nf_heavy > 0
            Sf_heavy = WilsonFermionAction(
                U,
                p.mass_heavy;
                Nf=p.Nf_heavy,
                csw=p.csw,
                anti_periodic=p.anti_periodic,
                cg_tol=p.cg_tol,
                cg_maxiters=p.cg_maxiters,
            )
            fermion_actions = (Sf_light, Sf_heavy)
        else
            fermion_actions = (Sf_light,)
        end
    elseif fermion_action == "staggered"
        Sf_light = StaggeredFermionAction(
            U,
            p.mass_light;
            Nf=p.Nf_light,
            anti_periodic=p.anti_periodic,
            cg_tol=p.cg_tol,
            cg_maxiters=p.cg_maxiters,
        )
        if p.Nf_heavy > 0
            Sf_heavy = StaggeredFermionAction(
                U,
                p.mass_heavy;
                Nf=p.Nf_heavy,
                anti_periodic=p.anti_periodic,
                cg_tol=p.cg_tol,
                cg_maxiters=p.cg_maxiters,
            )
            fermion_actions = (Sf_light, Sf_heavy)
        else
            fermion_actions = (Sf_light,)
        end
    else
        error("Fermion action \"$(fermion_action)\" not supported")
    end
    return fermion_actions
end

end
