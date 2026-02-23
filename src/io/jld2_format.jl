function save_field(
    ::JLD2Format, U::Gaugefield{B,T,false}, filename::String, args...
) where {B,T}
    filename != "" || return nothing

    if B == CPU
        jldsave(filename; U=Array(U.U))
    else
        tmp = convert_field(CPU, U, Float64)
        jldsave(filename; U=Array(tmp.U))
    end

    return nothing
end

function load_field!(::JLD2Format, U::Gaugefield{CPU,T,false}, filename::String) where {T}
    Unew = jldopen(filename, "r") do file
        file["U"]
    end
    @assert (size(Unew) == size(U.U)) "Size of supplied config is wrong"

    parallelfor(allindices(U), CPU, Val(false), (), (U,), (U,)) do μsite, (U,)
        U[μsite] = SMatrix{3,3,Complex{T},9}(Unew[μsite])
    end

    return nothing
end

function load_field!(::JLD2Format, U::Gaugefield{B,T,false,GA}, filename) where {B,T,GA}
    Ucpu = Gaugefield{CPU,Float64,GA,18}(size(U)..., U.β)
    load_field!(JLD2Format(), Ucpu, filename)
    Ugpu = convert_field(B, Ucpu, T)
    copy!(U, Ugpu)
    return nothing
end

function create_checkpoint(
    ::JLD2Format, univ, updatemethod, updatemethod_pt, itrj::Int, filename::String
)
    Uout = if univ.U isa Vector
       [convert_field(CPU, univ.U[i]) for i in eachindex(univ.U)]
    else
       convert_field(CPU, univ.U)
    end

    biasout = if univ.bias isa Vector
        [univ.bias[i].bias for i in eachindex(univ.bias)]
    elseif univ.bias == NoBias()
        NoBias()
    else
        univ.bias.bias
    end

    Pout = if updatemethod isa HMC
        convert_field(CPU, updatemethod.P)
    else
        nothing
    end

    state = get_rng_state()

    if filename != ""
        redirect_stderr(devnull) do
            jldsave(
                filename; 
                U=Uout,
                P=Pout,
                bias=biasout,
                numinstances=univ.numinstances,
                itrj=itrj,
                rngstate=state,
            )
        end
    end

    return nothing
end

function load_checkpoint(
    ::JLD2Format, parameters; rank=mpi_myrank(), mpi_multi_sim=false, build=false
)
    checkpoint_path = parameters.load_checkpoint_path
    instance = MPI_INSTANCE[]
    filename = joinpath(checkpoint_path, "checkpoint_$(instance)_$(rank).jld2")
    backend = parameters.backend
    B = BACKENDS[backend]
    T = Utils.FLOAT_TYPE[parameters.float_type]
    # TODO: support case of single node PT-MetaD
    U, _P, _bias, numinst, itrj, rngstate = jldopen(filename, "r") do file
        # INFO: versions older than 2.3.0 didnt checkpoint the momentum in HMC
        p = try
            file["P"]
        catch _
            nothing
        end

        convert_field(B, file["U"], T), p, file["bias"],
        file["numinstances"], file["itrj"], file["rngstate"]
    end

    dummy = parameters.tempering_enabled && mpi_multi_sim ? (instance==0) : false
    bias = if _bias == NoBias()
        NoBias()
    else
        Bias(parameters, U; bias=_bias, dummy, mpi_multi_sim, build)
    end
    recalc_cv!(U, bias)
    faction = init_fermion_actions(parameters, U)
    updatemethod = Updatemethod(parameters, U)

    if !isnothing(_P)
        P = convert_field(B, _P, T)
        copy!(updatemethod.P, P)
    end

    updatemethod_pt = nothing # TODO: support this case (serialize HMC and make method that takes P_old and U only)
    copy!(Random.default_rng(), rngstate)
    return U, faction, bias, numinst, updatemethod, updatemethod_pt, itrj
end
