# include("../src/system/MetaQCD.jl")
# using .MetaQCD
using Random

Random.seed!(1206)

NX = 8; NY = 8; NZ = 8; NT = 8;
U = random_gauges(NX, NY, NZ, NT, 5.7, type_of_gaction = WilsonGaugeAction);

smearing = StoutSmearing(U, 5, 0.125);
calc_smearedU!(smearing, U);

fully_smeared_U = smearing.Usmeared_multi[end];

staples = Temporaryfield(U);
dSdU = Temporaryfield(U);
temp_force = Temporaryfield(U);
MetaQCD.AbstractUpdateModule.calc_dSdU_bare!(dSdU, temp_force, staples, U, smearing);
# MetaQCD.AbstractSmearingModule.stout_backprop!(dSdU, temp_force, smearing);

calc_gauge_action(U)
calc_gauge_action(fully_smeared_U)