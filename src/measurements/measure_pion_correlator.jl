struct PionCorrelatorMeasurement{T,TD,TF,N} <: AbstractMeasurement
    dirac_operator::TD
    # eo_precon::Bool
    # mass_precon::Bool
    cg_tol::Float64
    cg_maxiters::Int64
    temp_fermion::TF # We need 1 temp fermion field for propagators
    temp_cg_fermions::NTuple{N,TF} # We need 4 temp fermions for cg / 7 for bicg(stab)
    pion_dict::Dict{Int64,Float64} # One value per time slice
    fp::T # file pointer
    function PionCorrelatorMeasurement(
        U::Gaugefield;
        filename="",
        printvalues=false,
        dirac_type="wilson",
        flow=false,
        mass=0.1,
        # Nf=2,
        csw=0,
        r=1,
        cg_tol=1e-16,
        cg_maxiters=1000,
        anti_periodic=true,
    )
        pion_dict = Dict{Int64,Float64}()
        NT = dims(U)[end]

        for it in 1:NT
            pion_dict[it] = 0.0
        end

        if dirac_type == "staggered"
            dirac_operator = StaggeredDiracOperator(U, mass; anti_periodic=anti_periodic)
            temp_fermion = Fermionfield(U; staggered=true)
            N = 6
            temp_cg_fermions = ntuple(_ -> Fermionfield(temp_fermion), 6)
        elseif dirac_type == "wilson"
            dirac_operator = WilsonDiracOperator(
                U, mass; anti_periodic=anti_periodic, r=r, csw=csw
            )
            temp_fermion = Fermionfield(U)
            N = 6
            temp_cg_fermions = ntuple(_ -> Fermionfield(temp_fermion), 6)
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

            for it in 1:NT
                header *= @sprintf("\t%-22s", "pion_corr_$(it)")
            end

            println(fp, header)
        else
            fp = nothing
        end

        return new{typeof(fp),typeof(dirac_operator),typeof(temp_fermion),N}(
            dirac_operator,
            cg_tol,
            cg_maxiters,
            temp_fermion,
            temp_cg_fermions,
            pion_dict,
            fp,
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
        # Nf=params.Nf,
        # κ=params.κ,
        # r=params.r,
        cg_tol=params.cg_tol,
        cg_maxiters=params.cg_maxiters,
        anti_periodic=params.anti_periodic,
    )
end

function measure(m::PionCorrelatorMeasurement{T}, U; additional_string="") where {T}
    measurestring = ""
    printstring = @sprintf("%-9s", additional_string)

    pion_correlators_avg!(
        m.pion_dict,
        m.dirac_operator(U),
        m.temp_fermion,
        m.temp_cg_fermions,
        m.cg_tol,
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

"""
    pion_correlators_avg!(dict, D, ψ, cg_temps, cg_tol, cg_maxiters)

Calculate the pion correlators for a given configuration and store the result for each
time slice in `dict`. \\
We follow the procedure outlined in DOI: 10.1007/978-3-642-01850-3 (Gattringer) pages
135-136 using point sources for each dirac and color index all starting from the origin
"""
function pion_correlators_avg!(dict, D, ψ, cg_temps, cg_tol, cg_maxiters)
    check_dims(D.U, ψ, cg_temps...)
    NX, NY, NZ, NT = dims(ψ)
    @assert length(dict) == NT
    source = SiteCoords(1, 1, 1, 1)
    propagator, temps... = cg_temps

    for a in 1:ψ.NC
        for μ in 1:ψ.ND
            ones!(propagator)
            set_source!(ψ, source, a, μ)
            solve_dirac!(propagator, D, ψ, temps...; tol=cg_tol, maxiters=cg_maxiters)
            for it in 1:NT
                cit = 0.0
                @batch reduction = (+, cit) for iz in 1:NZ
                    for iy in 1:NY
                        for ix in 1:NX
                            cit += real(
                                cdot(propagator[ix, iy, iz, it], propagator[ix, iy, iz, it])
                            )
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
