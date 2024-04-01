struct PionCorrelatorMeasurement{T,TD,TF} <: AbstractMeasurement
    dirac_operator::TD
    cg_tolerance::Float64
    cg_maxiters::Int64
    temp_fermion::Vector{TF} # We need 3 temp fermion fields for propagators
    temp_cg_fermions::Vector{TF} # We need 3 temp fermion fields for cg
    pion_dict::Dict{Int64,Float64} # One value per time slice
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
        pion_dict = Dict{Int64,Float64}()
        NT = dims(U)[end]

        for it in 1:NT
            pion_dict[it] = 0.0
        end

        if dirac_type == "staggered"
            dirac_operator = StaggeredDiracOperator(U, mass)
            temp_fermion = Fermionfield(U; staggered=true)
            temp_cg_fermions = [Fermionfield(U; staggered=true) for _ in 1:3]
        elseif dirac_type == "wilson"
            dirac_operator = WilsonDiracOperator(U, mass)
            temp_fermion = Fermionfield(U)
            temp_cg_fermions = [Fermionfield(U) for _ in 1:3]
        else
            throw(ArgumentError("Dirac operator \"$dirac_type\" is not supported"))
        end

        if printvalues
            fp = open(filename, "w")

            if flow
                str = @sprintf("%-9s\t%-7s\t%-9s", "itrj", "iflow", "tflow")
                println(fp, str)
            else
                str = @sprintf("%-9s", "itrj")
                println(fp, str)
            end

            for it in 1:NT
                str = @sprintf("\t%-22s", "pion_corr_$(it)")
                println(fp, str)
            end

            println(fp)
        else
            fp = nothing
        end

        return new{typeof(fp)}(
            dirac_operator, cg_tolerance, cg_maxiters, temp_fermion, temp_cg_fermions, fp
        )
    end
end

function PionCorrelatorMeasurement(
    U, params::PionCorrelatorParameters, filename, flow=false
)
    return PionCorrelatorMeasurement(
        U;
        filename=filename,
        printvalues=true,
        flow=flow,
        dirac_type=params.dirac_type,
        mass=params.mass,
        Nf=params.Nf,
        κ=params.κ,
        r=params.r,
        cg_tolerance=params.cg_tolerance,
        cg_maxiters=params.cg_maxiters,
        boundary_conditions=params.boundary_conditions,
    )
end

function measure(m::PionCorrelatorMeasurement{T}, U, ψ; additional_string="") where {T}
    measurestring = ""
    printstring = @sprintf("%-9s", additional_string)
    m.dirac_operator.U = U

    pion_correlators_avg!(
        m.pion_dict,
        m.dirac_operator,
        m.temp_fermion,
        m.temp_cg_fermions[1],
        m.temp_cg_fermions[2],
        m.temp_cg_fermions[3],
        m.temp_cg_fermions[4],
        m.cg_tolerance,
        m.cg_maxiters,
    )

    if T ≡ IOStream
        for value in values(m.pion_dict)
            svalue = @sprintf("%+-22.15E", value)
            printstring *= "\t$svalue"
        end

        measurestring = printstring
        println(m.fp, measurestring)
        flush(m.fp)
        measurestring *= " # pion_correlator"
    end

    output = MeasurementOutput(m.pion_dict, measurestring)
    return output
end

function pion_correlators_avg!(
    dict, D, ψ, temp1, temp2, temp3, temp4, cg_tolerance, cg_maxiters
)
    @assert dims(ψ) == dims(temp1) == dims(temp2) == dims(temp3) == dims(temp4) == dims(D.U)
    NX, NY, NZ, NT = dims(ψ)
    @assert length(dict) == NT
    source = SideCoords(1, 1, 1, 1)

    for a in 1:ψ.NC
        for μ in 1:ψ.ND
            clear!(temp1)
            set_source!(ψ, source, a, μ)
            solve_D⁻¹x!(temp1, D, ψ, temp2, temp3, temp4, cg_tolerance, cg_maxiters)
            for it in 1:NT
                cit = 0.0
                @batch reduction = (+, cit) for iz in 1:NZ
                    for iy in 1:NY
                        for ix in 1:NX
                            cit += cdot(temp1[ix, iy, iz, it], temp1[ix, iy, iz, it])
                        end
                    end
                end
                dict[it] = cit
            end
        end
    end

    Λₛ = NX * NY * NZ
    for it in 1:NT
        dict[it] /= Λₛ
    end

    return nothing
end
