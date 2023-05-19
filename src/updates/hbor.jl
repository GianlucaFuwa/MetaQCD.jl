struct HBORUpdate <: AbstractUpdate
    MAXIT::Int64
    prefactorHB::Float64
    prefactorOR::Float64
    numHB::Int64
    numOR::Int64
    _temporary_for_staples::TemporaryField

    function HBORUpdate(U, MAXIT, numHB, numOR)
        _temporary_for_staples = TemporaryField(U)

        return new(
            MAXIT, 
            U.NC / U.β, 
            U.β / U.NC, 
            numHB, 
            numOR,
            _temporary_for_staples,
        )
    end
end

function update!(
    updatemethod::HBORUpdate,
    U::Gaugefield,
    rng,
    verbose::VerboseLevel;
    metro_test::Bool = true,
)
    
    for i in 1:updatemethod.numHB
        heatbath_sweep!(
            U,
            updatemethod._temporary_for_staples,
            updatemethod.MAXIT,
            updatemethod.prefactorHB,
            rng,
        )
    end

    numaccepts = 0

    for j in 1:updatemethod.numOR
        numaccepts += OR_sweep!(
            U,
            updatemethod._temporary_for_staples,
            updatemethod.prefactorOR,
            rng,
        )
    end

    return numaccepts / (U.NV * 4.0 * updatemethod.numOR)
end