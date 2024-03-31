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
export BridgeFormat, JLD2Format, loadU!, saveU
export MetaLogger, current_time, @level1, @level2, @level3, set_global_logger!
export run_sim, run_build

import .BiasModule: Bias, Metadynamics, OPES, Parametric, calc_CV, update_bias!
import .Gaugefields: CPU, CUDABackend, ROCBackend
import .Gaugefields: DBW2GaugeAction, IwasakiGaugeAction, SymanzikTadGaugeAction
import .Gaugefields: SymanzikTreeGaugeAction, WilsonGaugeAction, Plaquette, Clover
import .Gaugefields: CoeffField, Gaugefield, Temporaryfield, initial_gauges
import .Gaugefields: calc_gauge_action, fieldstrength_eachsite!, identity_gauges!, mul!
import .Gaugefields: normalize!, plaquette, plaquette_trace_sum, random_gauges!
import .Gaugefields: substitute_U!, staple, staple_eachsite!, wilsonloop, to_backend
import .Gaugefields: Tensorfield, calc_kinetic_energy, gaussian_TA!
import .Measurements: measure, get_value, top_charge
import .Measurements: EnergyDensityMeasurement, GaugeActionMeasurement, PlaquetteMeasurement
import .Measurements: PolyakovMeasurement, TopologicalChargeMeasurement, WilsonLoopMeasurement
import .Parameters: ParameterSet, construct_params_from_toml
import .Smearing: Euler, RK2, RK3, RK3W7, GradientFlow, NoSmearing, StoutSmearing
import .Smearing: calc_smearedU!, flow!, stout_backprop!
import .Updates: Updatemethod, Heatbath, HMC, Metropolis
import .Updates: Leapfrog, OMF2, OMF2Slow, OMF4, OMF4Slow
import .Updates: Overrelaxation, Subgroups, KenneyLaub
import .Updates: calc_dSdU_bare!, calc_dVdU_bare!, evolve!, update!, ∇trFμνFρσ
import .Universe: Univ

export Bias, Metadynamics, OPES, Parametric, calc_CV, update_bias!
export CPU, CUDABackend, ROCBackend
export DBW2GaugeAction, IwasakiGaugeAction, SymanzikTadGaugeAction
export SymanzikTreeGaugeAction, WilsonGaugeAction, Plaquette, Clover
export CoeffField, Gaugefield, Temporaryfield, initial_gauges
export calc_gauge_action, fieldstrength_eachsite!, identity_gauges!, mul!
export normalize!, plaquette, plaquette_trace_sum, random_gauges!
export substitute_U!, staple, staple_eachsite!, wilsonloop
export Tensorfield, calc_kinetic_energy, gaussian_TA!
export measure, get_value, top_charge
export EnergyDensityMeasurement, GaugeActionMeasurement, PlaquetteMeasurement
export PolyakovMeasurement, TopologicalChargeMeasurement, WilsonLoopMeasurement
export ParameterSet, construct_params_from_toml
export Euler, RK2, RK3, RK3W7, GradientFlow, NoSmearing, StoutSmearing
export calc_smearedU!, flow!, stout_backprop!
export Updatemethod, Heatbath, HMC, Metropolis
export Leapfrog, OMF2, OMF2Slow, OMF4, OMF4Slow
export Overrelaxation, Subgroups, KenneyLaub
export calc_dSdU_bare!, calc_dVdU_bare!, evolve!, update!, ∇trFμνFρσ
export NoSmearing, StoutSmearing, calc_smearedU!, GradientFlow, flow!, stout_backprop!
export Updatemethod, update!, calc_dSdU_bare!, calc_dVdU_bare!, ∇trFμνFρσ
export Univ

end
