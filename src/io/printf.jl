# INFO: we need to define our own fopen and fclose using ccall because llvmcall is buggy:
# see: https://github.com/brenhinkeller/StaticTools.jl/issues/61
# But this doesn't work on Windows...
struct FILE end

@inline printfmt(::Type{<:AbstractFloat}) = "%g"
@inline printfmt(::Type{<:Integer}) = "%d"
@inline printfmt(::Type{UInt64}) = "%#x"
@inline printfmt(::Type{UInt32}) = "%#x"
@inline printfmt(::Type{<:AbstractString}) = "%s"

if Sys.iswindows() # ccall printf with floats doesnt work on windows for some reason
    using Format: cfmt
    @inline fopen(name::AbstractString, mode::AbstractString) = open(name, mode)
    @inline fclose(fp::IOStream) = close(fp)

    @inline printf(s) = print(cfmt("%s", s))
    @inline printf(n::T) where {T<:Number} = print(cfmt(printfmt(T), n))
    @inline printf(::Nothing) = nothing
    @inline printf(fp::IOStream, s) = print(fp, cfmt("%s", s))
    @inline printf(fp::IOStream, n::T) where {T<:Number} = print(fp, cfmt(printfmt(T), n))
    @inline printf(::IOStream, ::Nothing) = nothing
    @inline printf(fmt, s) = print(cfmt(fmt, s))
    @inline printf(fmt, n::Number) = print(cfmt(fmt, n))
    @inline printf(::AbstractString, ::Nothing) = nothing
    @inline printf(fp::IOStream, fmt, s) = print(fp, cfmt(fmt, s))
    @inline printf(fp::IOStream, fmt, n::Number) = print(fp, cfmt(fmt, n))
    @inline newline(fp) = println(fp)

    # Tuple
    @generated function printf(args::Tuple{Vararg{Any,N}}) where {N}
        return quote
            $(Expr(:meta, :inline))
            Base.Cartesian.@nexprs $N i->printf(args[i])
            return nothing
        end
    end

    @generated function printf(fp::IOStream, args::Tuple{Vararg{Any,N}}) where {N}
        return quote
            $(Expr(:meta, :inline))
            Base.Cartesian.@nexprs $N i->printf(fp, args[i])
            return nothing
        end
    end
else
    @inline function fopen(name::AbstractString, mode::AbstractString)
        GC.@preserve name mode fopen(pointer(name), pointer(mode))
    end

    @inline function fopen(name::Ptr{UInt8}, mode::Ptr{UInt8})
        ccall(:fopen, Ptr{FILE}, (Ptr{UInt8}, Ptr{UInt8}), name, mode)
    end

    @inline function fclose(fp::Ptr{FILE})
        if fp == C_NULL
            Int32(-1)
        else
            ccall(:fclose, Cint, (Ptr{FILE},), fp)
        end
    end

    macro printf(args...)
        return :(printf($(esc(args...))))
    end

    @inline printf(s) = GC.@preserve s printf(pointer(s))
    @inline printf(n::T) where {T<:Number} = printf(printfmt(T), n)
    @inline printf(::Nothing) = Int32(0)
    @inline printf(fp::Ptr{FILE}, s) = GC.@preserve s printf(fp, pointer(s))
    @inline printf(fp::Ptr{FILE}, n::T) where {T<:Number} = printf(fp, printfmt(T), n)
    @inline printf(::Ptr{FILE}, ::Nothing) = Int32(0)
    @inline printf(fmt, s) = GC.@preserve s printf(pointer(fmt), pointer(s))
    @inline printf(fmt, n::Number) = printf(pointer(fmt), n)
    @inline printf(::AbstractString, ::Nothing) = Int32(0)
    @inline printf(fp::Ptr{FILE}, fmt, s) = GC.@preserve fmt s printf(fp, pointer(fmt), pointer(s))
    @inline printf(fp::Ptr{FILE}, fmt, n::Number) = GC.@preserve fmt printf(fp, pointer(fmt), n)
    @inline newline(fp::Ptr{FILE}) = printf(fp, "\n")

    # StaticString
    @inline function printf(s::Ptr{UInt8})
        ccall(:printf, Cint, (Ptr{UInt8},), s)
    end

    @inline function printf(fp::Ptr{FILE}, s::Ptr{UInt8})
        ccall(:fprintf, Cint, (Ptr{FILE}, Ptr{UInt8}), fp, s)
    end

    @inline function printf(fmt::Ptr{UInt8}, s::Ptr{UInt8})
        ccall(:printf, Cint, (Ptr{UInt8}, Ptr{UInt8}), fmt, s)
    end

    @inline function printf(fp::Ptr{FILE}, fmt::Ptr{UInt8}, s::Ptr{UInt8})
        ccall(:fprintf, Cint, (Ptr{FILE}, Ptr{UInt8}, Ptr{UInt8}), fp, fmt, s)
    end

    # Int64 / UInt64
    @inline function printf(fmt::Ptr{UInt8}, n::Int64)
        ccall(:printf, Cint, (Ptr{UInt8}, Clonglong), fmt, n)
    end

    @inline function printf(fmt::Ptr{UInt8}, n::UInt64)
        ccall(:printf, Cint, (Ptr{UInt8}, Culonglong), fmt, n)
    end

    @inline function printf(fp::Ptr{FILE}, fmt::Ptr{UInt8}, n::Int64)
        ccall(:fprintf, Cint, (Ptr{FILE}, Ptr{UInt8}, Clonglong), fp, fmt, n)
    end

    @inline function printf(fp::Ptr{FILE}, fmt::Ptr{UInt8}, n::UInt64)
        ccall(:fprintf, Cint, (Ptr{FILE}, Ptr{UInt8}, Culonglong), fp, fmt, n)
    end

    # AbstractFloat
    @inline function printf(fmt::Ptr{UInt8}, n::AbstractFloat)
        ccall(:printf, Cint, (Ptr{UInt8}, Cdouble), fmt, Float64(n))
    end

    @inline function printf(fp::Ptr{FILE}, fmt::Ptr{UInt8}, n::AbstractFloat)
        ccall(:fprintf, Cint, (Ptr{FILE}, Ptr{UInt8}, Cdouble), fp, fmt, Float64(n))
    end

    # Tuple
    @generated function printf(args::Tuple{Vararg{Any,N}}) where {N}
        return quote
            $(Expr(:meta, :inline))
            Base.Cartesian.@nexprs $N i->printf(args[i])
            return zero(Int32)
        end
    end

    @generated function printf(fp::Ptr{FILE}, args::Tuple{Vararg{Any,N}}) where {N}
        return quote
            $(Expr(:meta, :inline))
            Base.Cartesian.@nexprs $N i->printf(fp, args[i])
            return zero(Int32)
        end
    end
end
