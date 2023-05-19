using Random

physical["L"] = (4,4,4,4)
physical["β"] = 5.0

meta["meta_enabled"] = false
meta["CVlims"] = (-10,10)
meta["CVthr"] = (-9,9)
meta["δq"] = 1e-3
meta["w"] = 1e-4
meta["k"] = 1000

sim["Ntherm"] = 1000
sim["Nsweeps"] = 10000
sim["initial"] = "hot"

mc["update_method"] = "HMC"
mc["ϵ_hmc"] = 0.0002
mc["hmc_steps"] = 100
                            
system["logdir"] = "./logs"
system["logfile"] = "L$(physical["L"])_beta$(physical["β"])_Qmax$(meta["Qmax"])_Qthr$(meta["Qthr"])_dq$(meta["δq"])_w$(meta["w"])_build.txt"
system["measure_dir"] = "./measurements/L$(physical["L"])_beta$(physical["β"])_Qmax$(meta["Qmax"])_Qthr$(meta["Qthr"])_dq$(meta["δq"])_w$(meta["w"])_build"
system["savebias_dir"] = "./metapotentials"
system["biasfile"] = "./L$(physical["L"])_beta$(physical["β"])_Qmax$(meta["Qmax"])_Qthr$(meta["Qthr"])_dq$(meta["δq"])_w$(meta["w"])_build.txt"
