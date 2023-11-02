module MetaQCD

using Requires

include("./utils/Utils.jl")
include("./output/Output.jl")
include("./fields/gauge/Gaugefields.jl")
include("./smearing/Smearing.jl")
include("./measurements/Measurements.jl")
include("./parameters/Parameters.jl")
include("./bias/Bias.jl")
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

export Sequential, SequentialMT, Checkerboard2, Checkerboard2MT, Checkerboard4, Checkerboard4MT
export loadU_bridge!, loadU_jld!, saveU_bridge, saveU_jld
export Verbose1, Verbose2, Verbose3
export run_sim, run_build

import .BiasModule: Bias, Metadynamics, OPES, Parametric, calc_CV, update_bias!
import .Gaugefields: DBW2GaugeAction, IwasakiGaugeAction, SymanzikTadGaugeAction
import .Gaugefields: SymanzikTreeGaugeAction, WilsonGaugeAction, Plaquette, Clover
import .Gaugefields: CoeffField, Gaugefield, Temporaryfield
import .Gaugefields: calc_gauge_action, fieldstrength_eachsite!, identity_gauges, mul!
import .Gaugefields: normalize!, plaquette, plaquette_trace_sum, random_gauges
import .Gaugefields: substitute_U!, staple, staple_eachsite!, wilsonloop
import .Gaugefields: Liefield, calc_kinetic_energy, gaussian_momenta!
import .Measurements: measure, get_value, top_charge
import .Measurements: GaugeActionMeasurement, PlaquetteMeasurement, PolyakovMeasurement
import .Measurements: TopologicalChargeMeasurement, WilsonLoopMeasurement
import .Parameters: ParameterSet, construct_params_from_toml
import .Smearing: Euler, RK2, RK3, RK3W7, GradientFlow, NoSmearing, StoutSmearing
import .Smearing: calc_smearedU!, flow!, stout_backprop!
import .Updates: Updatemethod, Heatbath, HMC, Metropolis
import .Updates: Overrelaxation, Subgroups, KenneyLaub
import .Updates: calc_dSdU_bare!, calc_dVdU_bare!, evolve!, update!, ∇trFμνFρσ
import .Universe: Univ

export Bias, Metadynamics, OPES, Parametric, calc_CV, update_bias!
export DBW2GaugeAction, IwasakiGaugeAction, SymanzikTadGaugeAction
export SymanzikTreeGaugeAction, WilsonGaugeAction, Plaquette, Clover
export CoeffField, Gaugefield, Temporaryfield
export calc_gauge_action, fieldstrength_eachsite!, identity_gauges, mul!
export normalize!, plaquette, plaquette_trace_sum, random_gauges
export substitute_U!, staple, staple_eachsite!, wilsonloop
export Liefield, calc_kinetic_energy, gaussian_momenta!
export measure, get_value, top_charge
export GaugeActionMeasurement, PlaquetteMeasurement, PolyakovMeasurement
export TopologicalChargeMeasurement, WilsonLoopMeasurement
export ParameterSet, construct_params_from_toml
export Euler, RK2, RK3, RK3W7, GradientFlow, NoSmearing, StoutSmearing
export calc_smearedU!, flow!, stout_backprop!
export Updatemethod, Heatbath, HMC, Metropolis
export Overrelaxation, Subgroups, KenneyLaub
export calc_dSdU_bare!, calc_dVdU_bare!, evolve!, update!, ∇trFμνFρσ
export NoSmearing, StoutSmearing, calc_smearedU!, GradientFlow, flow!, stout_backprop!
export Updatemethod, update!, calc_dSdU_bare!, calc_dVdU_bare!, ∇trFμνFρσ
export Univ

end
