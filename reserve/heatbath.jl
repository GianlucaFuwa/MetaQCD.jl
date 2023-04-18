module Heatbath
    
    import ..Utils: 
    import ..Gaugefields: Gaugefield

    struct Heatbath_update <: AbstractUpdate
        temp_U::Gaugefield
        ITERATION_MAX::Int64
        
        function Heatbath_update(U::Gaugefield;ITERATION_MAX = 10^5)
            temp_U = similar(U)
            return new(temp_U,ITERATION_MAX)
        end
    end

    const heatbath_factor = 2

    function heatbath_update_eachsite_SU2(
        A,
        μ,
        U::Gaugefield,
        h::Heatbath_update,
        
    )