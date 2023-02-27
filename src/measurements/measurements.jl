module Measurements

	import ..System_parameters:Params
	import ..System_parameters:set_params
    import ..Gaugefields:Gaugefield,calc_plaq,calc_Sgwils,top_charge
    import ..Metadynamics:Bias_potential,penalty_potential
    import ..Stout_smearing:stout_smear!

    export measurements,build_measurements,calc_topcharge,calc_weights

    defaultmeasures = Array{Dict,1}(undef,2)
    for i=1:length(defaultmeasures)
        defaultmeasures[i] = Dict()
    end
	
	defaultmeasures[1]["methodname"] = "Continuous_charge"
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
                elseif method["methodname"] == "Continuous_charge"
                    meas_files[i] = open(measure_dir*"/Continuous_charge.txt",meas_overwrite)
                elseif method["methodname"] == "Topological_charge"
                    meas_files[i] = open(measure_dir*"/Topological_charge.txt",meas_overwrite)
                elseif method["methodname"] == "Topological_susceptibility"
                    meas_files[i] = open(measure_dir*"/Topological_susceptibility.txt",meas_overwrite)
                else 
                    error("$(method["methodname"]) is not supported")
                end
            end
            return new(nummeas,meas_calls,meas_files)
        end
    end

    function measurements(itr,gfield::Gaugefield,gfieldcopy1::Gaugefield,gfieldcopy2::Gaugefield,measset::Measurement_set,numsmear::Int64)
        str = numsmear==0 ? itr : ""
        for iflow=1:numsmear+1
            for i = 1:measset.nummeas
                method = measset.meas_calls[i]
                measfile = measset.meas_files[i]
                if method["methodname"] == "Plaquette"
                    plaq = calc_plaq(gfieldcopy1)
                    plaq /= gfieldcopy1.NV
                    if iflow == 1 && numsmear > 1
                    print(measfile,itr," ",real(plaq)," ",imag(plaq))
                    elseif iflow == numsmear+1
                    println(measfile,str,"  "," ",real(plaq)," ",imag(plaq)," #itr Re_plaq Im_plaq (smear = 0:numsmear)")
                    else
                    print(measfile,"  "," ",real(plaq)," ",imag(plaq))
                    end
                elseif method["methodname"] == "Action"
                    s = gfieldcopy1.Sg/gfield.NV
                    if iflow == 1 && numsmear > 1
                    print(measfile,itr," ",s)
                    elseif iflow == numsmear
                    println(measfile,str," "," ",s," #itr action (smear = 0:numsmear)")
                    else
                    print(measfile," "," ",s)
                    end
                elseif method["methodname"] == "Meta_charge"
                    q = gfield.CV
                    if iflow == 1 && numsmear > 1
                    print(measfile,itr," ",q)
                    elseif iflow == numsmear
                    println(measfile,str," "," ",q," #itr metacharge (smear = 0:numsmear)")
                    else
                    print(measfile," "," ",q)
                    end
                elseif method["methodname"] == "Topological_charge"
                    q = top_charge(gfieldcopy1)
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
            stout_smear!(gfieldcopy1,gfieldcopy2,ρ_stout)
        end
        return nothing
    end

    function build_measurements(itr,gfield::Gaugefield,buildmeasures::Measurement_set)
        for i = 1:buildmeasures.nummeas
            method = buildmeasures.meas_calls[i]
            measfile = buildmeasures.meas_files[i]
            if itr % method["measure_every"] == 0
                if method["methodname"] == "Continuous_charge"
                    cv = gfield.CV
                    println(measfile,"$itr $cv # continuous charge")
                elseif method["methodname"] == "Action"
                    s = s = calc_Sgwils(gfield)/gfield.NV
                    println(measfile,"$itr $s # action")
                end
                flush(measfile)
            end
        end
        return nothing
    end

    function calc_weights(q_vals::Array{Float64,1},b::Bias_potential)
        @inline index(q,qmin,dq) = round(Int,(q-qmin)/dq+0.5,RoundNearestTiesAway)
        weights = zeros(length(q_vals))
        for i=1:length(q_vals)
            idx = index(q_vals[i],b.Qmin,b.δq)
            weights[i] = exp(b[idx]+penalty_potential(q_vals[i],b.Qmin_thr,b.Qmax_thr,b.k))
        end
        return weights
    end
    
end