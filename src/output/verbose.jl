import InteractiveUtils

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

const GlobalLogger = Ref(MetaLogger(2))

Base.flush(logger::MetaLogger) = flush(logger.io)
Base.close(logger::MetaLogger) = close(logger.io)

function set_global_logger!(level, io=devnull; tc=true)
    GlobalLogger[] = MetaLogger(level, io, tc)
    return nothing
end

for input_level in 1:3
    # Create the macros @level1, @level2, and @level3
    @eval macro $(Symbol("level$(input_level)"))(val...)
        return $(Symbol("level$(input_level)"))(val...)
    end
end

for input_level in 1:3
    # Create the functions that the macros @level1, @level2, and @level3 call
    @eval function $(Symbol("level$(input_level)"))(val...)
        GlobalLogger[].LEVEL < $(input_level) && return nothing
        return quote
            GlobalLogger[].to_console && println(stdout, $(esc(val...)))
            println(GlobalLogger[].io, $(esc(val...)))
            flush(GlobalLogger[].io)
        end
    end
end
