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
        TD = typeof(D)

        if Nf == default_Nf(D)
            if D isa StaggeredEOPreDiracOperator
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
                rhmc_info_action = nothing
                rhmc_info_md = nothing
                rhmc_temps1 = nothing
                rhmc_temps2 = nothing
            end

            R = false
            cg_temps = ntuple(_ -> Spinorfield(f; staggered=is_staggered(D)), 4)
        else
            @assert 1 <= Nf < default_Nf(D) """
            Nf should be between 1 or $(default_Nf(D)) for $(type) (was $Nf).
            If you want Nf > $(default_Nf(D)), use multiple actions
            """
            R = true
            rhmc_lambda_low = rhmc_spectral_bound[1]
            rhmc_lambda_high = rhmc_spectral_bound[2]
            cg_temps = ntuple(_ -> Spinorfield(f; staggered=is_staggered(D)), 2)
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
                _ -> Spinorfield(f; staggered=is_staggered(D)), n_temps + 1
            )
            rhmc_temps2 = ntuple(
                _ -> Spinorfield(f; staggered=is_staggered(D)), n_temps + 1
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
        |    BOUNDARY CONDITION (TIME): $(S.boundary_condition))
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

