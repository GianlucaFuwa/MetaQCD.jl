struct PionCorrelatorMeasurement{T,TD,TF} <: AbstractMeasurement
    dirac_operator::TD
    temp_fermions::Vector{TF} # We need 3 temp fermion fields for cg
    fp::T # file pointer
    function PionCorrelatorMeasurement(
        U::Gaugefield;
        filename="",
        printvalues=false,
        dirac_type="staggered",
        flow=false,
        mass=0.1,
        Nf=2,
        κ=1,
        r=1,
        cg_tolerance=1e-12,
        cg_maxiters=1000,
        boundary_conditions="periodic",
    )
        if dirac_type == "staggered"
            D = StaggeredDiracOperator(U, mass)
            temp_fermions = [Fermionfield(U; staggered=true) for i in 1:3]
        elseif dirac_type == "wilson"
            D = WilsonDiracOperator(U, mass)
            temp_fermions = [Fermionfield(U) for i in 1:3]
        else
            throw(ArgumentError("Dirac operator \"$dirac_type\" is not supported"))
        end

        if printvalues
            fp = open(filename, "w")
            header = ""
            if flow
                header *= @sprintf("%-9s\t%-7s\t%-9s", "itrj", "iflow", "tflow")
            else
                header *= @sprintf("%-9s", "itrj")
            end

            for methodname in TC_methods
                header *= "\tQ$(rpad(methodname, 22, " "))"
            end

            println(fp, header)
        else
            fp = nothing
        end

        return new{typeof(fp)}(TC_dict, fp)
    end
end

function PionCorrelatorMeasurement(
    U, params::PionCorrelatorParameters, filename, flow=false
)
    return PionCorrelatorMeasurement(
        U;
        filename=filename,
        printvalues=true,
        TC_methods=params.kinds_of_topological_charge,
        flow=flow,
    )
end

function measure(m::PionCorrelatorMeasurement{T}, U; additional_string="") where {T}
    measurestring = ""
    printstring = @sprintf("%-9s", additional_string)

    for methodname in keys(m.TC_dict)
        Q = top_charge(U, methodname)
        m.TC_dict[methodname] = Q
    end

    if T ≡ IOStream
        for value in values(m.TC_dict)
            svalue = @sprintf("%+-22.15E", value)
            printstring *= "\t$svalue"
        end

        measurestring = printstring
        println(m.fp, measurestring)
        flush(m.fp)
        measurestring *= " # top_charge"
    end

    output = MeasurementOutput(m.TC_dict, measurestring)
    return output
end

function pion_correlators_temporal!(
    correlators, D, ψ, temp1, temp2, temp3, temp4, cg_tolerance, cg_maxiters
)
    NX, NY, NZ, NT = dims(U)
    @assert length(correlators) == NT
    ones!(ψ)

    solve_D⁻¹x!(temps1, D, ψ, temps2, temps3, temp4, cg_tolerance, cg_maxiters)

    for it in 1:NT
        # -½Tr[ΓD⁻¹(0|m)ΓD⁻¹(m|0)]
    end

    return nothing
end
