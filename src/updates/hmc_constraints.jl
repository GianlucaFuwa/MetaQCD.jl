struct NoConstraint end

mutable struct HMCConstraint{D,TI,TS}
    info::TI
    smearing::TS
    name::String
    value::Float64
    variance::Float64
    function HMCConstraint(U, name, value, variance; numsmear=0, rho=0)
        if isnothing(name) || name == ""
            return NoConstraint()
        end

        @assert variance >= 0 "Constraint variance has to be >= 0"
        info = get_cvinfo_from_parameters((kind_of_cv=name,))
        smearing = StoutSmearing(U; numlayers=numsmear, rho)
        D = variance==0
        TI = typeof(info)
        TS = typeof(smearing)
        return new{D,TI,TS}(info, smearing, name, value, variance)
    end
end

constraint_is_delta(constraint::NoConstraint) = false
constraint_is_delta(constraint::HMCConstraint{D}) where {D} = D

calc_constraint_action(U, constraint::HMCConstraint{true}) = 0.0

function calc_constraint_action(U, constraint::HMCConstraint{false})
    C = calc_cv(U, constraint)[1]
    σ = constraint.variance
    μ = constraint.value
    return 1/2σ^2 * (μ - C)^2
end

function calc_cv(U, constraint::HMCConstraint{D,TI,TS}, args...) where {D,TI,TS<:NoSmearing}
    return (constraint.info.cv_func(U),)
end

function calc_cv(U, constraint::HMCConstraint{D,TI,TS}, args...) where {D,TI,TS}
    calc_smearedU!(constraint.smearing, U)
    levels = constraint.smearing.numlayers
    smeared_U = constraint.smearing.Usmeared_multi[levels+1]
    return (constraint.info.cv_func(smeared_U),)
end

function Base.show(io::IO, ::MIME"text/plain", c::HMCConstraint)
    str = """
        |  HMCConstraint(
        |    name: $(c.name)
        |    value: $(c.value)
        |    variance: $(c.variance)
        |    smearing: $(c.smearing)
        |  )"""
    return print(io, str)
end

function Base.show(io::IO, c::HMCConstraint)
    str = """
        |  HMCConstraint(
        |    name: $(c.name)
        |    value: $(c.value)
        |    variance: $(c.variance)
        |    smearing: $(c.smearing)
        |  )"""
    print(io, str)
    return nothing
end
