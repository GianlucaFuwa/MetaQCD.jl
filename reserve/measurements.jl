module Measurements

	import ..System_parameters: Params
    import ..Gaugefields: Gaugefield,substitute_U!,plaquette_trsum,calc_GaugeAction
    import ..Observables: top_charge,polyakov_tr
    import ..Metadynamics: Bias_potential,ReturnPotential
    import ..Stout_smearing: stout_smear!

    defaultmeasures = Array{Dict,1}(undef,2)
    for i=1:length(defaultmeasures)
        defaultmeasures[i] = Dict()
    end
	
	defaultmeasures[1]["methodname"] = "Meta_charge"
    defaultmeasures[1]["measure_every"] = 1
    defaultmeasures[2]["methodname"] = "Action"
    defaultmeasures[2]["measure_every"] = 1

    struct Measurement_set
        nummeas::Int64
        meas_calls::Array{Dict,1}
        meas_files::Array{IOStream,1}

        function Measurement_set(measure_dir;meas_calls=defaultmeasures)
            nummeas = length(meas_calls)
            meas_files = Array{IOStream,1}(undef,nummeas)
            for i=1:nummeas
                method = meas_calls[i]
                
                meas_overwrite = "w"
                if method["methodname"] == "Plaquette"
                    meas_files[i] = open(measure_dir*"/Plaquette.txt",meas_overwrite)
                elseif method["methodname"] == "Action"
                    meas_files[i] = open(measure_dir*"/Action.txt",meas_overwrite)
                elseif method["methodname"] == "Meta_charge"
                    meas_files[i] = open(measure_dir*"/Meta_charge.txt",meas_overwrite)
                elseif method["methodname"] == "Topological_charge"
                    meas_files[i] = open(measure_dir*"/Topological_charge.txt",meas_overwrite)
                else 
                    error("$(method["methodname"]) is not supported")
                end
            end
            return new(nummeas,meas_calls,meas_files)
        end
    end

    function measurements(itr,U::Gaugefield,Usmr::NTuple{2,Gaugefield},measset::Measurement_set,numsmear::Int64)
        str = numsmear==0 ? itr : ""
        if itr%measset.measure_every == 0
            substitute_U!(Usmr[1],U)
            substitute_U!(Usmr[1],U)
            for iflow=1:numsmear+1
                for i = 1:measset.nummeas
                    method = measset.meas_calls[i]
                    measfile = measset.meas_files[i]
                    if method["methodname"] == "Plaquette"
                        plaq = plaquette_trsum(Usmr)
                        plaq /= Ucopy1.NV
                        if iflow == 1 && numsmear > 1
                        print(measfile,itr," ",real(plaq)," ",imag(plaq))
                        elseif iflow == numsmear+1
                        println(measfile,str,"  "," ",real(plaq)," ",imag(plaq)," #itr Re_plaq Im_plaq (smear = 0:numsmear)")
                        else
                        print(measfile,"  "," ",real(plaq)," ",imag(plaq))
                        end
                    elseif method["methodname"] == "Action"
                        s = Usmr[1].Sg/UNV
                        if iflow == 1 && numsmear > 1
                        print(measfile,itr," ",s)
                        elseif iflow == numsmear
                        println(measfile,str," "," ",s," #itr action (smear = 0:numsmear)")
                        else
                        print(measfile," "," ",s)
                        end
                    elseif method["methodname"] == "Meta_charge"
                        q = Usmr.CV
                        if iflow == 1 && numsmear > 1
                        print(measfile,itr," ",q)
                        elseif iflow == numsmear
                        println(measfile,str," "," ",q," #itr metacharge (smear = 0:numsmear)")
                        else
                        print(measfile," "," ",q)
                        end
                    elseif method["methodname"] == "Topological_charge"
                        q = top_charge(Usmr[1])
                        if iflow == 1 && numsmear > 1
                        print(measfile,itr," ",q)
                        elseif iflow == numsmear
                        println(measfile,str," "," ",q," #itr topcharge (smear = 0:numsmear)")
                        else
                        print(measfile," "," ",q)
                        end
                    else 
                        error("$(method["methodname"]) is not supported")
                    end
                    flush(measfile)
                end
                stout_smear!(Usmr[1],Usmr[2],ρ_stout)
            end
        end
        return nothing
    end

    function calc_weights(q_vals::Array{Float64,1},b::Bias_potential)
        weights = zeros(length(q_vals))
        for (i,q) in enumerate(q_vals)
            if b.CVlims[1] <= q < b.CVlims[2]
                V = ReturnPotential(b,q)
                weights[i] = exp(V)
            else 
                weights[i] = 0.0
            end
        end
        return weights
    end
    
end