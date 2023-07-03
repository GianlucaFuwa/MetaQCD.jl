module MetaQCD
    using Requires

    include("./utilities/utils.jl")
    include("./output/verbose.jl")
    include("./output/io.jl")
    include("./fields/gauge/gaugefields.jl")
    include("./smearing/AbstractSmearing.jl")
    include("./measurements/AbstractMeasurement.jl")
    include("./system/parameter_structs.jl")
    include("./system/system_parameters.jl")
    include("./system/parameters_TOML.jl")
    include("./metadynamics/metadynamics.jl")
    include("./updates/tempering.jl")
    include("./system/universe.jl")
    include("./updates/AbstractUpdate.jl")
    include("./measurements/MeasurementMethods.jl")

    function __init__()
        @require MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195" begin
            MPI.Init()
        end
    end

    include("./system/mainrun.jl")
    include("./system/mainbuild.jl")

    using .IOModule
    using .Utils
    using .VerbosePrint

    import .SystemParameters: Params
    import .ParametersTOML: construct_params_from_toml
    import .Gaugefields: DBW2GaugeAction, IwasakiGaugeAction, SymanzikTadGaugeAction
    import .Gaugefields: SymanzikTreeGaugeAction, WilsonGaugeAction
    import .Gaugefields: CoeffField, Gaugefield, Temporaryfield, staple_eachsite!
    import .Gaugefields: calc_gauge_action, plaquette, plaquette_trace_sum
    import .Gaugefields: identity_gauges, random_gauges, normalize!, wilsonloop
    import .Gaugefields: calc_kinetic_energy, gaussian_momenta!, Liefield
    import .AbstractSmearingModule: NoSmearing, StoutSmearing, calc_smearedU!
    import .AbstractSmearingModule: GradientFlow, flow!
    import .AbstractMeasurementModule: measure, get_value, PlaquetteMeasurement, top_charge
    import .AbstractMeasurementModule: PolyakovMeasurement, WilsonLoopMeasurement
    import .AbstractMeasurementModule: TopologicalChargeMeasurement, GaugeActionMeasurement
    import .AbstractMeasurementModule: MetaChargeMeasurement
    import .Metadynamics: BiasPotential, update_bias!
    import .AbstractUpdateModule: update!, Updatemethod
    import .UniverseModule: Univ
    import .Mainrun: run_sim
    import .Mainbuild: run_build

    export construct_params_from_toml
    export loadU_bridge!, loadU_jld!, saveU_bridge, saveU_jld
    export Verbose1, Verbose2, Verbose3
    export DBW2GaugeAction, IwasakiGaugeAction, SymanzikTadGaugeAction
    export SymanzikTreeGaugeAction, WilsonGaugeAction
    export CoeffField, Gaugefield, Temporaryfield, calc_gauge_action
    export plaquette, wilsonloop, identity_gauges, random_gauges, move, normalize!
    export plaquette_trace_sum, staple_eachsite!
    export Liefield, gaussian_momenta!, calc_kinetic_energy
    export BiasPotential, update_bias!, return_potential
    export NoSmearing, StoutSmearing, calc_smearedU!, GradientFlow, flow!
    export Updatemethod, update!
    export measure, get_value, PlaquetteMeasurement, PolyakovMeasurement
    export WilsonLoopMeasurement, TopologicalChargeMeasurement, GaugeActionMeasurement
    export top_charge
    export Params, construct_measurement_parameters_from_toml, construct_params_from_toml
    export Univ
    export run_sim, run_build

end
