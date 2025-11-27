struct FermionAction{R,Nf,TD,TDl,CT,CTl,RI1,RI2,SA,SMD,TX} <: AbstractFermionAction{R,Nf}
    D::TD
    D_low::TDl
    temps::CT
    temps_low::CTl
    rhmc_info_action::RI1
    rhmc_info_md::RI2
    solver_action::SA
    solver_md::SMD
    Xμν::TX # Some actions need extra buffers/fields
    function FermionAction(
        type,
        f::AbstractField{B,T},
        mass;
        precon="none",
        # twisted_mass=Float64[],
        bc_str="antiperiodic",
        Nf=default_Nf(type),
        rhmc_spectral_bound=(minimum(mass)^2, 6.0),
        rhmc_order_action=15,
        rhmc_order_md=15,
        rhmc_prec_action=64,
        rhmc_prec_md=64,
        rhmc_tol_action=1e-8,
        rhmc_tol_md=1e-4,
        cg_tol_action=1e-7,
        cg_tol_md=1e-6,
        cg_maxiters_action=1000,
        cg_maxiters_md=1000,
        cg_float_type=T,
        cg_filepath="",
        kwargs...,
    ) where {B,T}
        D = DIRAC_OPERATORS[type](f, minimum(mass); bc_str=bc_str, kwargs...)
        temp = D.temp
        eo_fun = contains(type, "eo") ? even_odd : identity
        TD = typeof(D)
        Tsolve = cg_float_type
        is_mixed = Tsolve != T

        if Nf == default_Nf(D)
            if D isa StaggeredEOPreDiracOperator
                power = Nf//2default_Nf(D)
                rhmc_info_action = RHMCParams(
                    power;
                    n_max=rhmc_order_action,
                    lambda_low=rhmc_spectral_bound[1],
                    lambda_high=rhmc_spectral_bound[2],
                    precision=rhmc_prec_action,
                )
                @assert all(rhmc_info_action.maxerr .<= rhmc_tol_action) """
                Rational approximation max. error for action is above given \"rhmc_tol_action\":
                tol: $(rhmc_tol_action)
                maxerr: $(rhmc_info_action.maxerr) (positive and negative power)
                """
                n_temps = max(get_n(rhmc_info_action), get_n_inverse(rhmc_info_action))
                temps = ntuple(_ -> even_odd(Spinorfield(temp)), 2n_temps + 2 + 4)
                U_low = is_mixed ? Gaugefield(f, Tsolve) : nothing
                if is_mixed
                    n_temps = max(get_n(rhmc_info_action), get_n_inverse(rhmc_info_action))
                    # ntemps+1 for solution vectors, 2 for Ap & r
                    temps = ntuple(_ -> even_odd(Spinorfield(temp)), n_temps + 3) 
                    D_low = DIRAC_OPERATORS[type](U_low, minimum(mass); bc_str=bc_str, kwargs...)
                    # ntemps+1 for solution vectors, ntemps+1 for gradients, 3 for Ap & r & r_old
                    temps_low = (U_low, ntuple(_ -> even_odd(Spinorfield(temp, Tsolve; staggered=true)), 2n_temps + 2 + 3)...)
                else
                    # ntemps+1 for solution vectors, ntemps+1 for gradient vectors, 2 for Ap & r
                    n_temps = max(get_n(rhmc_info_action), get_n_inverse(rhmc_info_action))
                    temps = ntuple(_ -> even_odd(Spinorfield(temp)), 2n_temps + 2 + 2) 
                    D_low = nothing
                    temps_low = nothing
                end
                rhmc_info_md = nothing
            else
                rhmc_info_action = nothing
                rhmc_info_md = nothing
                staggered = num_dirac(temp) == 1
                U_low = is_mixed ? Gaugefield(f, Tsolve) : nothing
                if is_mixed
                    temps = ntuple(_ -> eo_fun(Spinorfield(temp)), 3)
                    D_low = DIRAC_OPERATORS[type](U_low, minimum(mass); bc_str=bc_str, kwargs...)
                    temps_low = (U_low, ntuple(_ -> eo_fun(Spinorfield(temp, Tsolve; staggered)), 5)...)
                else
                    temps = ntuple(_ -> eo_fun(Spinorfield(temp)), 4)
                    D_low = nothing
                    temps_low = nothing
                end
            end

            R = false
        else
            @assert 1 <= Nf < default_Nf(D) """
            Nf should be between 1 or $(default_Nf(D)) for $(type) (was $Nf).
            If you want Nf > $(default_Nf(D)), use multiple actions
            """
            R = true
            rhmc_lambda_low = rhmc_spectral_bound[1]
            rhmc_lambda_high = rhmc_spectral_bound[2]

            fun = if precon == "heavy"
                @assert length(mass) == 2
                δm² = 4(maximum(mass)^2 - minimum(mass)^2)
                x -> (x + δm²) / x
            elseif precon == "none"
                @assert mass isa Float64
                x -> x
            else
                error("precon in fermion_action can only be \"heavy\" or \"none\"")
            end

            power = Nf//2default_Nf(D)
            rhmc_info_action = RHMCParams(
                power,
                fun;
                n_max=rhmc_order_action,
                lambda_low=rhmc_lambda_low,
                lambda_high=rhmc_lambda_high,
                precision=rhmc_prec_action,
            )
            @assert all(rhmc_info_action.maxerr .<= rhmc_tol_action) """
            Rational approximation max. error for action is above given \"rhmc_tol_action\":
            tol: $(rhmc_tol_action)
            maxerr: $(rhmc_info_action.maxerr) (positive and negative power)
                """
            power = Nf//default_Nf(D)
            rhmc_info_md = RHMCParams(
                power,
                fun;
                n_max=rhmc_order_md,
                lambda_low=rhmc_lambda_low,
                lambda_high=rhmc_lambda_high,
                precision=rhmc_prec_md,
            )
            @assert all(rhmc_info_md.maxerr .<= rhmc_tol_md) """
            Rational approximation max. error for md is above given \"rhmc_tol_md\":
            tol: $(rhmc_tol_md)
            maxerr: $(rhmc_info_md.maxerr) (positive and negative power)
            """
            n_temps_action = max(get_n(rhmc_info_action), get_n_inverse(rhmc_info_action))
            n_temps_md = max(get_n(rhmc_info_md), get_n_inverse(rhmc_info_md))
            n_temps = max(n_temps_action, n_temps_md)
            if is_mixed
                # ntemps+1 for solution vectors, 2 for Ap & r
                temps = ntuple(_ -> eo_fun(Spinorfield(temp)), n_temps + 3) 
                U_low = Gaugefield(f, Tsolve)
                D_low = DIRAC_OPERATORS[type](U_low, minimum(mass); bc_str=bc_str, kwargs...)
                # ntemps+1 for solution vectors, ntemps+1 for gradients, 3 for Ap & r & r_old
                temps_low = (U_low, ntuple(_ -> eo_fun(Spinorfield(temp, Tsolve; staggered=true)), 2n_temps + 2 + 3)...)
            else
                # ntemps+1 for solution vectors, ntemps+1 for gradient vectors, 2 for Ap & r
                temps = ntuple(_ -> eo_fun(Spinorfield(temp)), 2n_temps + 2 + 2) 
                U_low = nothing
                D_low = nothing
                temps_low = nothing
            end
        end

        Xμν = if type ∈ ("wilson", "wilson_eo")
            has_clover_term(D) ? Tensorfield(f) : nothing
        else
            nothing
        end

        solverfile_action = if cg_filepath == ""
            nothing
        else
            SStaticString(cg_filepath * "_action_$(lpad(MPI_INSTANCE[], 3, "0")).txt")
        end

        solverfile_md = if cg_filepath == ""
            nothing
        else
            SStaticString(cg_filepath * "_md_$(lpad(MPI_INSTANCE[], 3, "0")).txt")
        end

        if !isnothing(solverfile_action)
            fp = fopen(solverfile_action, "w")
            if is_mixed
                printf(fp, "%-11s", "outer iters")
                printf(fp, "%-11s", "inner iters")
            else
                printf(fp, "%-11s", "iters")
            end
            printf(fp, "%-25s", "res")
            printf(fp, "%-25s", "elapsed time [s]")
            newline(fp)
            fclose(fp)
        end

        if !isnothing(solverfile_md)
            fp = fopen(solverfile_md, "w")
            if is_mixed
                printf(fp, "%-11s", "outer iters")
                printf(fp, "%-11s", "inner iters")
            else
                printf(fp, "%-11s", "iters")
            end
            printf(fp, "%-25s", "res")
            printf(fp, "%-25s", "elapsed time [s]")
            newline(fp)
            fclose(fp)
        end

        solver_action = SolverInfo(cg!, cg_tol_action, cg_maxiters_action, solverfile_action)
        solver_md = SolverInfo(cg!, cg_tol_md, cg_maxiters_md, solverfile_md)

        TDl = typeof(D_low)
        CT = typeof(temps)
        CTl = typeof(temps_low)
        RI1 = typeof(rhmc_info_action)
        RI2 = typeof(rhmc_info_md)
        SA = typeof(solver_action)
        SMD = typeof(solver_md)
        TX = typeof(Xμν)
        return new{R,Nf,TD,TDl,CT,CTl,RI1,RI2,SA,SMD,TX}(
            D,
            D_low,
            temps,
            temps_low,
            rhmc_info_action,
            rhmc_info_md,
            solver_action,
            solver_md,
            Xμν,
        )
    end
end

function init_fermion_action(parameters, U, i)
    fparams = fermion_parameters_from_dict(parameters.fermions[i])
    cg_filepath = if mpi_amroot(MPI_COMM_INSTANCE[]) && (parameters.log_dir != "")
        joinpath(parameters.log_dir, "cg_data$(i)")
    else
        ""
    end

    action = FermionAction(
        parameters.fermion_action,
        U,
        fparams.mass;
        bc_str=parameters.boundary_condition,
        r=parameters.wilson_r,
        csw=parameters.wilson_csw,
        precon=fparams.precon,
        Nf=fparams.Nf,
        rhmc_spectral_bound=(fparams.rhmc_spectral_bound),
        rhmc_order_action=fparams.rhmc_order_action,
        rhmc_order_md=fparams.rhmc_order_md,
        rhmc_prec_action=fparams.rhmc_prec_action,
        rhmc_prec_md=fparams.rhmc_prec_md,
        rhmc_tol_action=fparams.rhmc_tol_action,
        rhmc_tol_md=fparams.rhmc_tol_md,
        cg_tol_action=fparams.cg_tol_action,
        cg_tol_md=fparams.cg_tol_md,
        cg_maxiters_action=fparams.cg_maxiters_action,
        cg_maxiters_md=fparams.cg_maxiters_md,
        cg_float_type=Utils.FLOAT_TYPE[parameters.solver_float_type],
        cg_filepath=cg_filepath,
    )
    return action
end

"""
    calc_fermion_action(fermion_action, U, ϕ)

Calculate the fermion action for the fermion field `ϕ` on the gauge background `U`using the
fermion action `fermion_action`.
"""
# TODO: fermion action for actions with 1 or 2 twisted masses
# TM = 0: No Twisted mass S = (ϕ, (D†D)⁻¹, ϕ)
# TM = 1: Twisted mass in Denominator S = (ϕ, (D†D + μ₀)⁻¹, ϕ)
# TM = 2: Twisted mass in Numerator S = (ϕ, (D†D + μ₀)(D†D)⁻¹, ϕ)
# TM = 3: Twisted mass in Numerator and Denominator S = (ϕ, (D†D + μ₀)(D†D + μ₁)⁻¹, ϕ)
function calc_fermion_action(
    fermion_action::AbstractFermionAction{false,Nf}, U::Gaugefield{B,T}, ϕ
) where {Nf,B,T}
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    solver_action = fermion_action.solver_action
    tol, maxiters, datafile = get_info(solver_action)

    if isnothing(fermion_action.D_low)
        ψ, temps... = fermion_action.temps[1:4]
        clear!(ψ) # initial guess is zero
        solve_dirac!(ψ, DdagD, ϕ, temps; tol, maxiters, datafile)
    else
        # TODO: delta = solver_action.delta
        ψ, temps... = fermion_action.temps[1:3]
        clear!(ψ) # initial guess is zero
        U_low, temps_low... = fermion_action.temps_low[1:6]
        copy!(U_low, U)
        D_low = fermion_action.D_low(U_low)
        DdagD_low = DdaggerD(D_low)
        solve_dirac_mixed!(
            ψ, DdagD, DdagD_low, ϕ, temps, temps_low; tol, maxiters, datafile
        )
    end

    Sf = real(dot(ϕ, ψ))
    return Sf
end

function calc_fermion_action(
    fermion_action::AbstractFermionAction{true,Nf}, U::Gaugefield{B,T}, ϕ
) where {Nf,B,T}
    rhmc = fermion_action.rhmc_info_action
    n = get_n_inverse(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    solver_action = fermion_action.solver_action
    tol, maxiters, datafile = get_info(solver_action)

    shifts = get_β_inverse(rhmc)
    coeffs = get_α_inverse(rhmc)
    α₀ = get_α0_inverse(rhmc)

    if isnothing(fermion_action.D_low)
        ψs = fermion_action.temps[1:n+1]
        ps = fermion_action.temps[n+2:2n+2]
        temps = fermion_action.temps[2n+3:2n+4]

        for v in ψs
            clear!(v)
        end

        solve_dirac_multishift!(
            ψs, shifts, DdagD, ϕ, temps, ps; tol, maxiters, datafile
        )
    else
        # TODO: delta = solver_action.delta
        ψs = fermion_action.temps[1:n+1]
        temps = fermion_action.temps[n+2:n+3]
        U_low = fermion_action.temps_low[1]
        ψs_low = fermion_action.temps_low[2:n+2]
        ps_low = fermion_action.temps_low[n+3:2n+3]
        temps_low = fermion_action.temps_low[2n+4:2n+6]
        copy!(U_low, U)
        D_low = fermion_action.D_low(U_low)
        DdagD_low = DdaggerD(D_low)

        for v in ψs
            clear!(v)
        end

        solve_dirac_multishift_mixed!(
            ψs, shifts, DdagD, DdagD_low, ϕ, temps,
            ψs_low, ps_low, temps_low;
            tol, maxiters, datafile
        )
    end

    ψ = ψs[1]
    clear!(ψ) # D⁻¹ϕ doesn't appear in the partial fraction decomp so we can use it to sum

    axpy!(α₀, ϕ, ψ)

    for i in 1:n
        axpy!(coeffs[i], ψs[i+1], ψ)
    end

    Sf = real(dot(ψ, ψ))
    return Sf
end

# TODO: fermion action for actions with 1 or 2 twisted masses
# TM = 0: No Twisted mass
# TM = 1: Only twisted mass in Numerator
# TM = 2: Twisted mass in Denominator
# TM = 3: Twisted mass in Numerator and Denominator
# function calc_fermion_action(fermion_action::AbstractFermionAction{false,Nf,1}, U, ϕ) where {Nf}
# end

calc_fermion_action(::QuenchedFermionAction, ::Gaugefield, ::Any) = 0.0

"""
    sample_pseudofermions!(ϕ, fermion_action, U)

Sample pseudo fermions for an HMC update according to the probability density specified by
`fermion_action`.
"""
function sample_pseudofermions!(ϕ, fermion_action::AbstractFermionAction{false}, U)
    D = fermion_action.D(U)
    temp = fermion_action.temps[1]
    gaussian_pseudofermions!(temp)
    LinearAlgebra.mul!(ϕ, adjoint(D), temp)
    return nothing
end

function sample_pseudofermions!(
    ϕ, fermion_action::F, U
) where {F<:Union{FermionAction{true},FermionAction{false,4,<:StaggeredEOPreDiracOperator}}}
    rhmc = fermion_action.rhmc_info_action
    n = get_n(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    solver_action = fermion_action.solver_action
    tol, maxiters, datafile = get_info(solver_action)

    shifts = get_β(rhmc)
    coeffs = get_α(rhmc)
    α₀ = get_α0(rhmc)
    gaussian_pseudofermions!(ϕ) # D⁻¹ϕ doesn't appear in the partial fraction decomp so we can use it to sum

    if isnothing(fermion_action.D_low)
        ψs = fermion_action.temps[1:n+1]
        ps = fermion_action.temps[n+2:2n+2]
        temps = fermion_action.temps[2n+3:2n+4]

        for v in ψs
            clear!(v)
        end

        solve_dirac_multishift!(
            ψs, shifts, DdagD, ϕ, temps, ps; tol, maxiters, datafile
        )
    else
        # TODO: delta = solver_action.delta
        ψs = fermion_action.temps[1:n+1]
        temps = fermion_action.temps[n+2:n+3]
        U_low = fermion_action.temps_low[1]
        ψs_low = fermion_action.temps_low[2:n+2]
        ps_low = fermion_action.temps_low[n+3:2n+3]
        temps_low = fermion_action.temps_low[2n+4:2n+6]
        copy!(U_low, U)
        D_low = fermion_action.D_low(U_low)
        DdagD_low = DdaggerD(D_low)

        for v in ψs
            clear!(v)
        end

        solve_dirac_multishift_mixed!(
            ψs, shifts, DdagD, DdagD_low, ϕ, temps,
            ψs_low, ps_low, temps_low;
            tol, maxiters, datafile
        )
    end

    mul!(ϕ, ϕ, α₀)

    for i in 1:n
        axpy!(coeffs[i], ψs[i+1], ϕ)
    end

    return nothing
end

sample_pseudofermions!(::AbstractField, ::QuenchedFermionAction, U) = nothing

### I/O stuff
function Base.show(io::IO, ::MIME"text/plain", S::QuenchedFermionAction)
    return println(io, "| Quenched")
end

function Base.show(io::IO, ::MIME"text/plain", S::FermionAction{R,Nf,TD}) where {R,Nf,TD}
    D = S.D
    name = chop(string(nameof(TD)); tail=length("DiracOperator")) * "FermionAction"
    print(
        io,
        """
        |  $(name)
        |    Nf: $Nf
        |    MASS: $(D.mass)""",
    )

    if D isa WilsonDiracOperator || D isa WilsonEOPreDiracOperator
        print(
            io,
            """
            |    KAPPA: $(D.κ)
            |    CSW: $(D.csw)""",
        )
    elseif D isa StaggeredHoelblingDiracOperator
        print(
            io,
            """
            |    MASS TERM: $(_unwrap_val.(get_mass_term(D)))""",
        )
    end

    print(
        io,
        """
        |    BOUNDARY CONDITION (TIME): $(nameof(typeof(D.boundary_condition)))
        |    CG TOLERANCE (ACTION): $(S.solver_action.tol)
        |    CG TOLERANCE (MD): $(S.solver_md.tol)
        |    CG MAX ITERS (ACTION): $(S.solver_action.maxiters)
        |    CG MAX ITERS (ACTION): $(S.solver_md.maxiters)
        |    RHMC INFO (Action): $(S.rhmc_info_action)
        |    RHMC INFO (MD): $(S.rhmc_info_md)""",
    )
    return nothing
end

function Base.show(io::IO, S::FermionAction{R,Nf,TD}) where {R,Nf,TD}
    D = S.D
    name = chop(string(nameof(TD)); tail=length("DiracOperator")) * "FermionAction"
    print(
        io,
        """
        |  $(name)
        |    Nf: $Nf
        |    MASS: $(D.mass)""",
    )

    if D isa WilsonDiracOperator || D isa WilsonEOPreDiracOperator
        print(
            io,
            """
            |    KAPPA: $(D.κ)
            |    CSW: $(D.csw)""",
        )
    elseif D isa StaggeredHoelblingDiracOperator
        print(
            io,
            """
            |    MASS TERM: $(_unwrap_val.(get_mass_term(D)))""",
        )
    end

    print(
        io,
        """
        |    BOUNDARY CONDITION (TIME): $(nameof(typeof(D.boundary_condition)))
        |    CG TOLERANCE (ACTION): $(S.solver_action.tol)
        |    CG TOLERANCE (MD): $(S.solver_md.tol)
        |    CG MAX ITERS (ACTION): $(S.solver_action.maxiters)
        |    CG MAX ITERS (MD): $(S.solver_md.maxiters)
        |    RHMC INFO (Action): $(S.rhmc_info_action)
        |    RHMC INFO (MD): $(S.rhmc_info_md)""",
    )
    return nothing
end
