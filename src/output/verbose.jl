using InteractiveUtils: InteractiveUtils
using MPI

MPI.Initialized() || MPI.Init()
const COMM = MPI.COMM_WORLD
const MYRANK = MPI.Comm_rank(COMM)

"""
    MetaLogger(LEVEL::Int64, io::IO, to_console::Bool)

Logger object that can be used to log messages to the IOStream `io` and, if
`to_console = true`, to the console.
"""
struct MetaLogger
    LEVEL::Int64
    io::IO
    to_console::Bool
    MetaLogger(LEVEL=2, tc::Bool=true) = new(LEVEL, devnull, tc)
    MetaLogger(LEVEL, ::Nothing, tc::Bool=true) = new(LEVEL, devnull, tc)
    MetaLogger(LEVEL, filename::String, tc::Bool=true) = new(LEVEL, open(filename, "w"), tc)
    MetaLogger(LEVEL, io::IO, tc::Bool=true) = new(LEVEL, io, tc)
end

const __GlobalLogger = Ref(MetaLogger(2))

Base.flush(logger::MetaLogger) = flush(logger.io)
Base.close(logger::MetaLogger) = close(logger.io)

function set_global_logger!(level, io=devnull; tc=true)
    __GlobalLogger[] = MetaLogger(level, io, tc)
    return nothing
end

# functions would be enough, but macros look cooler
# Could also define them using @eval as below, but then the LSP doesn't register them and
# we get "Missing reference" everywhere
macro level1(val...)
    return level1(val...)
end

macro level2(val...)
    return level2(val...)
end

macro level3(val...)
    return level3(val...)
end

for input_level in 1:3
    # Create the functions that the macros @level1, @level2, and @level3 call
    @eval function $(Symbol("level$(input_level)"))(val...)
        return quote
            if Output.__GlobalLogger[].LEVEL >= $($input_level) && MYRANK == 0
                Output.__GlobalLogger[].to_console && println(stdout, $(esc(val...)))
                println(Output.__GlobalLogger[].io, $(esc(val...)))
                flush(Output.__GlobalLogger[].io)
            end
        end
    end
end
