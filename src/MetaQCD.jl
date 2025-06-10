module MetaQCD

using Requires

include("./utils/Utils.jl")
include("./parameters/Parameters.jl")
include("./fields/Fields.jl")
include("./io/IO.jl")
include("./solvers/Solvers.jl")
include("./rhmc/AlgRemez.jl")
include("./rhmc/RHMCParameters.jl")
include("./smearing/Smearing.jl")
include("./diracoperators/DiracOperators.jl")
include("./measurements/Measurements.jl")
include("./bias/Bias.jl")
include("./forces/Forces.jl")
include("./main/Universe.jl")
include("./updates/Updates.jl")
include("./main/Main.jl")

using .MetaIO
using .Utils
using .Main
using .Solvers
using LinearAlgebra
using Unicode

export BMWFormat, BridgeFormat, JLD2Format, load_config!, save_config
export MetaLogger, current_time, @level1, @level2, @level3, set_global_logger!
export run_sim, build_bias

import .BiasModule: Bias, Metadynamics, NoBias, OPES, Parametric, calc_cv, update_bias!
import .DiracOperators: AbstractDiracOperator, Daggered, DdaggerD, calc_fermion_action
import .DiracOperators: StaggeredDiracOperator, StaggeredEOPreDiracOperator, even_odd
import .DiracOperators: WilsonDiracOperator, WilsonEOPreDiracOperator, sample_pseudofermions!
import .DiracOperators: StaggeredHoelblingDiracOperator, FermionAction
import .DiracOperators: QuenchedFermionAction
import .Fields: CPU, DBW2GaugeAction, IwasakiGaugeAction, SymanzikTadGaugeAction
import .Fields: SymanzikTreeGaugeAction, WilsonGaugeAction, Plaquette, Clover
import .Fields: Expfield, Colorfield, Gaugefield
import .Fields: calc_gauge_action, fieldstrength_eachsite!, identity_gauges!, to_backend
import .Fields: normalize!, plaquette, plaquette_trace_sum, random_gauges!
import .Fields: staple, staple_eachsite!, wilsonloop, to_backend
import .Fields: Tensorfield, calc_kinetic_energy, gaussian_TA!
import .Fields: Spinorfield, gaussian_pseudofermions!
import .Forces: calc_dSdU_bare!, calc_dSfdU_bare!, calc_dVdU_bare!
import .Measurements: measure, top_charge
import .Measurements: EnergyDensityMeasurement, GaugeActionMeasurement, PlaquetteMeasurement
import .Measurements: PolyakovMeasurement, TopologicalChargeMeasurement
import .Measurements: WilsonLoopMeasurement, ∇trFμνFρσ
import .Parameters: ParameterSet, construct_params_from_toml
import .Smearing: Euler, RK2, RK3, RK3W7, GradientFlow, NoSmearing, StoutSmearing
import .Smearing: calc_smearedU!, flow!, stout_backprop!
import .Updates: Updatemethod, Heatbath, HMC, Metropolis, evolve!, update!
import .Updates: Leapfrog, LeapfrogRA, OMF2, OMF2Slow, OMF4, OMF4RA, OMF4Slow
import .Updates: Overrelaxation, Subgroups, KenneyLaub
import .Universe: Univ

export Bias, Metadynamics, NoBias, OPES, Parametric, calc_cv, update_bias!
export CPU, DBW2GaugeAction, IwasakiGaugeAction, SymanzikTadGaugeAction
export SymanzikTreeGaugeAction, WilsonGaugeAction, Plaquette, Clover
export Expfield, Colorfield, Gaugefield
export calc_gauge_action, fieldstrength_eachsite!, identity_gauges!, to_backend
export normalize!, plaquette, plaquette_trace_sum, random_gauges!
export staple, staple_eachsite!, wilsonloop
export Tensorfield, calc_kinetic_energy, gaussian_TA!
export Spinorfield, Daggered, DdaggerD
export StaggeredDiracOperator, StaggeredEOPreDiracOperator 
export WilsonDiracOperator, WilsonEOPreDiracOperator, QuenchedFermionAction
export StaggeredHoelblingDiracOperator, FermionAction
export even_odd, sample_pseudofermions!
export calc_fermion_action, gaussian_pseudofermions!
export measure, top_charge
export EnergyDensityMeasurement, GaugeActionMeasurement, PlaquetteMeasurement
export PolyakovMeasurement, TopologicalChargeMeasurement, WilsonLoopMeasurement
export ParameterSet, construct_params_from_toml
export Euler, RK2, RK3, RK3W7, GradientFlow, NoSmearing, StoutSmearing
export calc_smearedU!, flow!, stout_backprop!
export Updatemethod, Heatbath, HMC, Metropolis
export Leapfrog, LeapfrogRA, OMF2, OMF2Slow, OMF4, OMF4RA, OMF4Slow
export Overrelaxation, Subgroups, KenneyLaub
export calc_dSdU_bare!, calc_dSfdU_bare!, calc_dVdU_bare!, evolve!, update!, ∇trFμνFρσ
export NoSmearing, StoutSmearing, calc_smearedU!, GradientFlow, flow!, stout_backprop!
export Updatemethod, update!
export Univ

# include("utils/precompile.jl")

end
