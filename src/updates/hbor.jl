struct HBOR_update <: AbstractUpdate
    MAXIT::Int64
    prefactorHB::Float64
    prefactorOR::Float64
    numHB::Int64
    numOR::Int64

    function HBOR_update(U, MAXIT, numHB, numOR)
        prefactorHB = U.NC / U.β
        prefactorOR = U.β / U.NC
        return new(
            MAXIT, 
            prefactorHB, 
            prefactorOR, 
            numHB, 
            numOR,
            )
    end
end

get_MAXIT(hbor::HBOR_update) = hbor.MAXIT
get_prefactorHB(hbor::HBOR_update) = hbor.prefactorHB
get_prefactorOR(hbor::HBOR_update) = hbor.prefactorOR
get_numHB(hbor::HBOR_update) = hbor.numHB
get_numOR(hbor::HBOR_update) = hbor.numOR

function update!(
    updatemethod::T,
    U::Gaugefield,
    rng::Xoshiro,
    verbose::Verbose_level;
    metro_test::Bool = true,
    ) where {T<:HBOR_update}
    
    MAXIT = get_MAXIT(updatemethod)
    prefactorHB = get_prefactorHB(updatemethod)
    prefactorOR = get_prefactorOR(updatemethod)

    for i = 1:get_numHB(updatemethod)
        heatbath_sweep!(U, MAXIT, prefactorHB, rng)
    end
    numaccepts = 0
    for j = 1:get_numOR(updatemethod)
        numaccepts += OR_sweep!(U, prefactorOR, rng)
    end
    recalc_GaugeAction!(U)

    return numaccepts / U.NV / 4.0
end