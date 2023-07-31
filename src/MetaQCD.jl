module MetaQCD
    using Requires

    include("./utils/Utils.jl")
    include("./output/Output.jl")
    include("./fields/gauge/Gaugefields.jl")
    include("./smearing/Smearing.jl")
    include("./measurements/Measurements.jl")
    include("./parameters/Parameters.jl")
    include("./metadynamics/Metadynamics.jl")
    include("./main/Universe.jl")
    include("./updates/Updates.jl")

    function __init__()
        @require MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195" begin
            MPI.Init()
        end
    end

    include("./main/Mainrun.jl")

    using .Output
    using .Utils
    using .Mainrun

    import .Gaugefields: DBW2GaugeAction, IwasakiGaugeAction, SymanzikTadGaugeAction
    import .Gaugefields: SymanzikTreeGaugeAction, WilsonGaugeAction
    import .Gaugefields: CoeffField, Gaugefield, Temporaryfield, staple_eachsite!
    import .Gaugefields: calc_gauge_action, plaquette, plaquette_trace_sum
    import .Gaugefields: identity_gauges, random_gauges, normalize!, wilsonloop
    import .Gaugefields: calc_kinetic_energy, gaussian_momenta!, Liefield
    import .Measurements: measure, get_value, PlaquetteMeasurement, top_charge
    import .Measurements: PolyakovMeasurement, WilsonLoopMeasurement
    import .Measurements: TopologicalChargeMeasurement, GaugeActionMeasurement
    import .Measurements: MetaChargeMeasurement
    import .Metadynamics: BiasPotential, update_bias!
    import .Parameters: ParameterSet, construct_params_from_toml
    import .Smearing: NoSmearing, StoutSmearing, GradientFlow, calc_smearedU!, flow!
    import .Updates: update!, Updatemethod
    import .Universe: Univ

    export construct_params_from_toml
    export loadU_bridge!, loadU_jld!, saveU_bridge, saveU_jld
    export Verbose1, Verbose2, Verbose3
    export DBW2GaugeAction, IwasakiGaugeAction, SymanzikTadGaugeAction
    export SymanzikTreeGaugeAction, WilsonGaugeAction
    export CoeffField, Gaugefield, Temporaryfield, calc_gauge_action
    export plaquette, wilsonloop, identity_gauges, random_gauges, move, normalize!
    export plaquette_trace_sum, staple_eachsite!
    export Liefield, gaussian_momenta!, calc_kinetic_energy
    export BiasPotential, update_bias!
    export NoSmearing, StoutSmearing, calc_smearedU!, GradientFlow, flow!
    export Updatemethod, update!
    export measure, get_value, PlaquetteMeasurement, PolyakovMeasurement
    export WilsonLoopMeasurement, TopologicalChargeMeasurement, GaugeActionMeasurement
    export top_charge
    export ParameterSet, construct_params_from_toml
    export Univ
    export run_sim, run_build

end
