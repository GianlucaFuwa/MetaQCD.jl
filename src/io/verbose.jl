"""
    MetaLogger(LEVEL::Int64, io::IO, to_console::Bool)

Logger object that can be used to log messages to the IOStream `io` and, if
`to_console = true`, to the console.
"""
struct MetaLogger
    LEVEL::Int64
    fp::Union{Nothing,Ptr{FILE},IOStream}
    to_console::Bool
    MetaLogger(LEVEL=2, tc::Bool=true) = new(LEVEL, nothing, tc)
    MetaLogger(LEVEL, ::Nothing, tc::Bool=true) = new(LEVEL, nothing, tc)
    MetaLogger(LEVEL, file::String, tc::Bool=true) = new(LEVEL, fopen(file, "w"), tc)
    MetaLogger(LEVEL, fp::Ptr{FILE}, tc::Bool=true) = new(LEVEL, fp, tc)
end

Base.close(logger::MetaLogger) = !isnothing(logger.fp) && fclose(logger.fp)

const __GlobalLogger = Ref(MetaLogger(2))

@inline prints_to_console() = __GlobalLogger[].to_console

function set_global_logger!(level, fp_or_file=nothing; tc=true)
    __GlobalLogger[] = MetaLogger(level, fp_or_file, tc)
    return nothing
end

printf(::Nothing, ::Any) = zero(Int32)

macro level1(msg)
    pmsg = prepare_message(msg)
    return quote
        if __GlobalLogger[].LEVEL ≥ 1 && mpi_amroot()
            __GlobalLogger[].to_console && printf($pmsg)
            !isnothing(__GlobalLogger[].fp) && printf(__GlobalLogger[].fp, $pmsg)
        end
        nothing
    end
end

macro level2(msg)
    pmsg = prepare_message(msg)
    return quote
        if __GlobalLogger[].LEVEL ≥ 2 && mpi_amroot()
            __GlobalLogger[].to_console && printf($pmsg)
            !isnothing(__GlobalLogger[].fp) && printf(__GlobalLogger[].fp, $pmsg)
        end
        nothing
    end
end

macro level3(msg)
    pmsg = prepare_message(msg)
    return quote
        if __GlobalLogger[].LEVEL ≥ 3 && mpi_amroot()
            __GlobalLogger[].to_console && printf($pmsg)
            !isnothing(__GlobalLogger[].fp) && printf(__GlobalLogger[].fp, $pmsg)
        end
        nothing
    end
end

@inline prepare_message(msg::String) = :(($msg, "\n"))

@inline prepare_message(msg::Symbol) = :(($(esc(msg)), "\n"))

@inline function prepare_message(msg::Expr)
    @assert msg.head == :string
    tup = Expr(:tuple)

    for arg in msg.args
        # if arg isa String
        #     push!(tup.args, :($arg))
        # else
            push!(tup.args, :($(esc(arg))))
        # end
    end

    push!(tup.args, :("\n"))
    return tup
end
