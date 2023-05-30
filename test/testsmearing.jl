include("../src/system/MetaQCD.jl")
using .MetaQCD
using Random

Random.seed!(1206)

NX = 4; NY = 4; NZ = 4; NT = 4;
U = random_gauges(NX, NY, NZ, NT, 5.7, gaction = "wilson");

smearing = StoutSmearing(U, 5, 0.125);
calc_smearedU!(smearing, U);

fully_smeared_U = smearing.Usmeared_multi[end];

staples = TemporaryField(U);
dSdU = TemporaryField(U);
temp_force = TemporaryField(U);
MetaQCD.AbstractUpdateModule.calc_dSdU_bare!(dSdU, temp_force, staples, U, smearing);
MetaQCD.AbstractSmearingModule.stout_backprop!(dSdU, temp_force, smearing);

calc_gauge_action(U)
calc_gauge_action(fully_smeared_U)