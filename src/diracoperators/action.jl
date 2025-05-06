struct FermionAction{R,Nf,TD,CT,RI1,RI2,RT,TX,T} <: AbstractFermionAction{R,Nf}
    D::TD
    cg_temps::CT
    rhmc_info_action::RI1
    rhmc_info_md::RI2
    rhmc_temps1::RT # this holds the results of multishift cg
    rhmc_temps2::RT # this holds the basis vectors in multishift cg
    cg_tol_action::Float64
    cg_tol_md::Float64
    cg_maxiters_action::Int64
    cg_maxiters_md::Int64
    cg_datafile::T
    Xμν::TX # Some actions need extra buffers/fields
    function FermionAction(
        type,
        f::AbstractField,
        mass;
        bc_str="antiperiodic",
        Nf=default_Nf(type),
        rhmc_spectral_bound=(mass^2, 6.0),
        rhmc_order_action=15,
        rhmc_prec_action=42,
        rhmc_order_md=10,
        rhmc_prec_md=42,
        cg_tol_action=1e-14,
        cg_tol_md=1e-12,
        cg_maxiters_action=1000,
        cg_maxiters_md=1000,
        cg_filepath="",
        kwargs...,
    )
        D = DIRAC_OPERATORS[type](f, mass; bc_str=bc_str, kwargs...)
        eo_fun = contains(type, "eo") ? even_odd : identity
        TD = typeof(D)

        if Nf == default_Nf(D)
            if D isa StaggeredEOPreDiracOperator
                cg_temps = ntuple(_ -> even_odd(Spinorfield(f; staggered=true)), 4)
                power = Nf//2default_Nf(D)
                rhmc_info_action = RHMCParams(
                    power;
                    n=rhmc_order_action,
                    precision=rhmc_prec_action,
                    lambda_low=rhmc_spectral_bound[1],
                    lambda_high=rhmc_spectral_bound[2],
                )
                n_temps = rhmc_order_action
                rhmc_temps1 = ntuple(
                    _ -> even_odd(Spinorfield(f; staggered=true)), n_temps + 1
                )
                rhmc_temps2 = ntuple(
                    _ -> even_odd(Spinorfield(f; staggered=true)), n_temps + 1
                )
                rhmc_info_md = nothing
            else
                cg_temps = ntuple(_ -> eo_fun(Spinorfield(f; staggered=is_staggered(D))), 4)
                rhmc_info_action = nothing
                rhmc_info_md = nothing
                rhmc_temps1 = nothing
                rhmc_temps2 = nothing
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
            cg_temps = ntuple(_ -> eo_fun(Spinorfield(f; staggered=is_staggered(D))), 2)
            power = Nf//2default_Nf(D)
            rhmc_info_action = RHMCParams(
                power;
                n=rhmc_order_action,
                precision=rhmc_prec_action,
                lambda_low=rhmc_lambda_low,
                lambda_high=rhmc_lambda_high,
            )
            power = Nf//default_Nf(D)
            rhmc_info_md = RHMCParams(
                power;
                n=rhmc_order_md,
                precision=rhmc_prec_md,
                lambda_low=rhmc_lambda_low,
                lambda_high=rhmc_lambda_high,
            )
            n_temps = max(rhmc_order_md, rhmc_order_action)
            rhmc_temps1 = ntuple(
                _ -> eo_fun(Spinorfield(f; staggered=is_staggered(D))), n_temps + 1
            )
            rhmc_temps2 = ntuple(
                _ -> eo_fun(Spinorfield(f; staggered=is_staggered(D))), n_temps + 1
            )
        end

        Xμν = if type ∈ ("wilson", "wilson_eo")
            has_clover_term(D) ? Tensorfield(f) : nothing
        else
            nothing
        end

        cg_datafile = StaticString(cg_filepath)

        if cg_filepath != ""
            open(cg_datafile, "w") do fp
                @printf(fp, "%-11s%-25s", "iters", "res")
                println(fp)
            end
        end

        CT = typeof(cg_temps)
        RI1 = typeof(rhmc_info_action)
        RI2 = typeof(rhmc_info_md)
        RT = typeof(rhmc_temps1)
        TX = typeof(Xμν)
        T = typeof(cg_datafile)
        return new{R,Nf,TD,CT,RI1,RI2,RT,TX,T}(
            D,
            cg_temps,
            rhmc_info_action,
            rhmc_info_md,
            rhmc_temps1,
            rhmc_temps2,
            cg_tol_action,
            cg_tol_md,
            cg_maxiters_action,
            cg_maxiters_md,
            cg_datafile,
            Xμν,
        )
    end
end

function init_fermion_action(params, mass::Float64, Nf::Int64, U)
    cg_filepath = if mpi_amroot(MPI_COMM_INSTANCE[]) && (params.log_dir != "")
        joinpath(params.log_dir, "cg_data_$(lpad(MPI_INSTANCE[], 3, "0")).txt")
    else
        ""
    end

    action = FermionAction(
        params.fermion_action, U, mass;
        bc_str=params.boundary_condition,
        Nf=Nf,
        rhmc_spectral_bound=(params.rhmc_spectral_bound),
        rhmc_order_md=params.rhmc_order_md,
        rhmc_prec_md=params.rhmc_prec_md,
        rhmc_order_action=params.rhmc_order_action,
        rhmc_prec_action=params.rhmc_prec_action,
        cg_tol_action=params.cg_tol_action,
        cg_tol_md=params.cg_tol_md,
        cg_maxiters_action=params.cg_maxiters_action,
        cg_maxiters_md=params.cg_maxiters_md,
        cg_filepath=cg_filepath,
        r=params.wilson_r,
        csw=params.wilson_csw,
    )
    return action
end

"""
    calc_fermion_action(fermion_action, U, ϕ)

Calculate the fermion action for the fermion field `ϕ` on the gauge background `U`using the
fermion action `fermion_action`.
"""
function calc_fermion_action(fermion_action::AbstractFermionAction{false}, U, ϕ)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    ψ, temp1, temp2, temp3 = fermion_action.cg_temps
    cg_tol = fermion_action.cg_tol_action
    cg_maxiters = fermion_action.cg_maxiters_action

    clear!(ψ) # initial guess is zero
    iters, res = solve_dirac!(ψ, DdagD, ϕ, temp1, temp2, temp3, cg_tol, cg_maxiters) # ψ = (D†D)⁻¹ϕ

    cg_datafile = fermion_action.cg_datafile
    if cg_datafile != ""
        set_ext!(cg_datafile, MPI_INSTANCE[])
        fp = fopen(cg_datafile, "a")
        printf(fp, "%-11i", iters)
        printf(fp, "%-25.15E", res)
        newline(fp)
        fclose(fp)
    end

    Sf = real(dot(ϕ, ψ))
    return Sf
end

function calc_fermion_action(fermion_action::AbstractFermionAction{true}, U, ϕ)
    cg_tol = fermion_action.cg_tol_action
    cg_maxiters = fermion_action.cg_maxiters_action
    rhmc = fermion_action.rhmc_info_action
    n = get_n(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    ψs = fermion_action.rhmc_temps1[1:n+1]
    ps = fermion_action.rhmc_temps2[1:n+1]
    temp1, temp2 = fermion_action.cg_temps

    for v in ψs
        clear!(v)
    end

    shifts = get_β_inverse(rhmc)
    coeffs = get_α_inverse(rhmc)
    α₀ = get_α0_inverse(rhmc)
    iters, res = solve_dirac_multishift!(ψs, shifts, DdagD, ϕ, temp1, temp2, ps, cg_tol, cg_maxiters)

    cg_datafile = fermion_action.cg_datafile
    if cg_datafile != ""
        set_ext!(cg_datafile, MPI_INSTANCE[])
        fp = fopen(cg_datafile, "a")
        printf(fp, "%-11i", iters)
        printf(fp, "%-25.15E", res)
        newline(fp)
        fclose(fp)
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

calc_fermion_action(::QuenchedFermionAction, ::Gaugefield, ::Any) = 0.0

"""
    sample_pseudofermions!(ϕ, fermion_action, U)

Sample pseudo fermions for an HMC update according to the probability density specified by
`fermion_action`.
"""
function sample_pseudofermions!(ϕ, fermion_action::AbstractFermionAction{false}, U)
    D = fermion_action.D(U)
    temp = fermion_action.cg_temps[1]
    gaussian_pseudofermions!(temp)
    LinearAlgebra.mul!(ϕ, adjoint(D), temp)
    return nothing
end

function sample_pseudofermions!(
    ϕ, fermion_action::F, U
) where {F<:Union{FermionAction{true},FermionAction{false,4,<:StaggeredEOPreDiracOperator}}}
    cg_tol = fermion_action.cg_tol_action
    cg_maxiters = fermion_action.cg_maxiters_action
    rhmc = fermion_action.rhmc_info_action
    n = get_n(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    ψs = fermion_action.rhmc_temps1[1:n+1]
    ps = fermion_action.rhmc_temps2[1:n+1]
    temp1, temp2 = fermion_action.cg_temps

    for v in ψs
        clear!(v)
    end

    shifts = get_β(rhmc)
    coeffs = get_α(rhmc)
    α₀ = get_α0(rhmc)
    gaussian_pseudofermions!(ϕ) # D⁻¹ϕ doesn't appear in the partial fraction decomp so we can use it to sum
    solve_dirac_multishift!(ψs, shifts, DdagD, ϕ, temp1, temp2, ps, cg_tol, cg_maxiters)

    mul!(ϕ, ϕ, α₀)

    for i in 1:n
        axpy!(coeffs[i], ψs[i+1], ϕ)
    end

    return nothing
end

sample_pseudofermions!(::AbstractField, ::QuenchedFermionAction, U) = nothing

### I/O stuff
function Base.show(io::IO, ::MIME"text/plain", S::QuenchedFermionAction)
    println(io, "| Quenched")
end

function Base.show(io::IO, ::MIME"text/plain", S::FermionAction{R,Nf,TD}) where {R,Nf,TD}
    D = S.D
    name = chop(string(nameof(TD)), tail=length("DiracOperator")) * "FermionAction"
    print(
        io,
        """

        |  $(name)(
        |    Nf: $Nf
        |    MASS: $(D.mass)
        """
    )

    if D isa WilsonDiracOperator || D isa WilsonEOPreDiracOperator
        print(
            io,
            """
            |    KAPPA: $(D.κ)
            |    CSW: $(D.csw)
            """
        )
    elseif D isa StaggeredHoelblingDiracOperator
        print(
            io,
            """
            |    MASS TERM: $(_unwrap_val.(get_mass_term(D)))
            """
        )
    end

    print(
        io,
        """
        |    BOUNDARY CONDITION (TIME): $(D.boundary_condition))
        |    CG TOLERANCE (ACTION): $(S.cg_tol_action)
        |    CG TOLERANCE (MD): $(S.cg_tol_md)
        |    CG MAX ITERS (ACTION): $(S.cg_maxiters_action)
        |    CG MAX ITERS (ACTION): $(S.cg_maxiters_md)
        |    RHMC INFO (Action): $(S.rhmc_info_action)
        |    RHMC INFO (MD): $(S.rhmc_info_md))
        """
    )
    return nothing
end

function Base.show(io::IO, S::FermionAction{R,Nf,TD}) where {R,Nf,TD}
    D = S.D
    name = chop(string(nameof(TD)), tail=length("DiracOperator")) * "FermionAction"
    print(
        io,
        """

        |  $(name)(
        |    Nf: $Nf
        |    MASS: $(D.mass)
        """
    )

    if D isa WilsonDiracOperator || D isa WilsonEOPreDiracOperator
        print(
            io,
            """
            |    KAPPA: $(D.κ)
            |    CSW: $(D.csw)
            """
        )
    elseif D isa StaggeredHoelblingDiracOperator

        print(
            io,
            """
            |    MASS TERM: $(_unwrap_val.(get_mass_term(D)))
            """
        )
    end

    print(
        io,
        """
        |    BOUNDARY CONDITION (TIME): $(D.boundary_condition))
        |    CG TOLERANCE (ACTION) = $(S.cg_tol_action)
        |    CG TOLERANCE (MD) = $(S.cg_tol_md)
        |    CG MAX ITERS (ACTION) = $(S.cg_maxiters_action)
        |    CG MAX ITERS (ACTION) = $(S.cg_maxiters_md)
        |    RHMC INFO (Action): $(S.rhmc_info_action)
        |    RHMC INFO (MD): $(S.rhmc_info_md))
        """
    )
    return nothing
end
