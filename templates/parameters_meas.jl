using Random

physical["L"] = (4,4,4,4)
physical["β"] = 5.0
physical["update_method"] = "HMC"

meta["Qmax"] = (-5000,5000)
meta["Qthr"] = (-2000,2000)
meta["δq"] = 10
meta["w"] = 1e-4
meta["k"] = 1000

sim["ϵ_metro"] = 0.1
sim["ϵ_hmc"] = 0.0002
sim["hmc_steps"] = 50
sim["Ntherm"] = 100
sim["Nsweeps"] = 100
sim["initial"] = "hot"
sim["parallel_tempering"] = true
sim["swap_every"] = 1

system["randomseeds"] = [Random.Xoshiro(1206),Random.Xoshiro(1209)]

system["meas_calls"] = Dict[Dict{Any,Any}("methodname" => "Continuous_charge","measure_every" => 10),
						    Dict{Any,Any}("methodname" => "Topological_charge","measure_every" => 10),
                            Dict{Any,Any}("methodname" => "Action","measure_every" => 10),
                            Dict{Any,Any}("methodname" => "Topological_susceptibility","measure_every" => 10)]

                            
system["logdir"] = "./logs"
system["logfile"] = "N$(physical["N"])_beta$(physical["β"])_Qmax$(meta["Qmax"])_Qthr$(meta["Qthr"])_dq$(meta["δq"])_w$(meta["w"])_k$(meta["k"])_TEMPER$(sim["swap_every"]).txt"
system["measure_dir"] = "./measurements/N$(physical["N"])_beta$(physical["β"])_Qmax$(meta["Qmax"])_Qthr$(meta["Qthr"])_dq$(meta["δq"])_w$(meta["w"])_k$(meta["k"])_TEMPER$(sim["swap_every"])"
system["savebias_dir"] = "./metapotentials"
system["biasfile"] = "N$(physical["N"])_beta$(physical["β"])_Qmax$(meta["Qmax"])_Qthr$(meta["Qthr"])_dq$(meta["δq"])_w$(meta["w"])_k$(meta["k"])_TEMPER$(sim["swap_every"]).txt"
system["usebias"] = "./metapotentials/N$(physical["N"])_beta$(physical["β"])_Qmax$(meta["Qmax"])_Qthr$(meta["Qthr"])_dq$(meta["δq"])_w$(meta["w"])_k$(meta["k"])_build.txt"
