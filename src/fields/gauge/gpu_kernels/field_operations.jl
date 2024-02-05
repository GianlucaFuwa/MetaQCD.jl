function substitute_U!(a::Abstractfield{GPUD,T}, b::Abstractfield{GPUD,T}) where {T}
	@assert size(b) == size(a)
	@latmap(substitute_U_kernel!, a, b)
	return nothing
end

@kernel function substitute_U_kernel!(a, @Const(b))
	site = @index(Global, Cartesian)

	@unroll for μ in 1i32:4i32
		@inbounds a[μ,site] = b[μ,site]
	end
end

clear_U!(u::Abstractfield{GPUD,T}) where {T} = @latmap(clear_U_kernel!, u, T)

@kernel function clear_U_kernel!(U, T)
	site = @index(Global, Cartesian)

	@unroll for μ in 1i32:4i32
		@inbounds U[μ,site] = zero3(T)
	end
end

normalize!(u::Abstractfield{GPUD}) = @latmap(normalize_kernel!, u)

@kernel function normalize_kernel!(U)
	site = @index(Global, Cartesian)

	@unroll for μ in 1i32:4i32
		@inbounds U[μ,site] = proj_onto_SU3(U[μ,site])
	end
end

function add!(a::Abstractfield{GPUD,T}, b::Abstractfield{GPUD,T}, fac) where {T}
	@assert size(b) == size(a)
	@latmap(add_kernel!, a, b, T(fac))
	return nothing
end

@kernel function add_kernel!(a, @Const(b), fac)
	site = @index(Global, Cartesian)

	@unroll for μ in 1i32:4i32
		@inbounds a[μ,site] = a[μ,site] + fac*b[μ,site]
	end
end

mul!(a::Abstractfield{GPUD,T}, α) where {T} = @latmap(mul_kernel!, a, T(α))

@kernel function mul_kernel!(a, α)
	site = @index(Global, Cartesian)

	@unroll for μ in 1i32:4i32
		@inbounds a[μ,site] = α*a[μ,site]
	end
end

function leftmul!(a::Abstractfield{GPUD,T}, b::Abstractfield{GPUD,T}) where {T}
	@assert size(b) == size(a)
	@latmap(leftmul_kernel!, a, b)
	return nothing
end

@kernel function leftmul_kernel!(a, @Const(b))
	site = @index(Global, Cartesian)

	@unroll for μ in 1i32:4i32
		@inbounds a[μ,site] = cmatmul_oo(b[μ,site], a[μ,site])
	end
end

function leftmul_dagg!(a::Abstractfield{GPUD,T}, b::Abstractfield{GPUD,T}) where {T}
	@assert size(b) == size(a)
	@latmap(leftmul_dagg_kernel!, a, b)
	return nothing
end

@kernel function leftmul_dagg_kernel!(a, @Const(b))
	site = @index(Global, Cartesian)

	@unroll for μ in 1i32:4i32
		@inbounds a[μ,site] = cmatmul_do(b[μ,site], a[μ,site])
	end
end

function rightmul!(a::Abstractfield{GPUD,T}, b::Abstractfield{GPUD,T}) where {T}
	@assert size(b) == size(a)
	@latmap(rightmul_kernel!, a, b)
	return nothing
end

@kernel function rightmul_kernel!(a, @Const(b))
	site = @index(Global, Cartesian)

	@unroll for μ in 1i32:4i32
		@inbounds a[μ,site] = cmatmul_oo(a[μ,site], b[μ,site])
	end
end

function rightmul_dagg!(a::Abstractfield{GPUD,T}, b::Abstractfield{GPUD,T}) where {T}
	@assert size(b) == size(a)
	@latmap(rightmul_dagg_kernel!, a, b)
	return nothing
end

@kernel function rightmul_dagg_kernel!(a, @Const(b))
	site = @index(Global, Cartesian)

	@unroll for μ in 1i32:4i32
		@inbounds a[μ,site] = cmatmul_od(a[μ,site], b[μ,site])
	end
end
