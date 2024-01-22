using Dates
using Logging
using LoggingExtras

struct MetaLogger
    LEVEL::Int64
    io::IO
    to_console::Bool

    MetaLogger(LEVEL=2) = new(LEVEL, devnull, true)
    MetaLogger(LEVEL, ::Nothing) = new(LEVEL, devnull, true)
    MetaLogger(LEVEL, filename::String) = new(LEVEL, open(filename, "w"), true)
    MetaLogger(LEVEL, io::IO) = new(LEVEL, io, true)
end

struct LogLevel{VAL}
    LogLevel(val::Integer) = new{val}()
end

const GlobalLogger = Ref(MetaLogger(2))

set_global_logger!(level, io=devnull) = GlobalLogger[]=MetaLogger(level, io)

for input_level in 1:3
    @eval macro $(Symbol("level$(input_level)"))(val...)
        return $(Symbol("level$(input_level)"))(val...)
    end
end

for input_level in 1:3
    @eval function $(Symbol("level$(input_level)"))(val...)
        GlobalLogger[].LEVEL<$(input_level) && return nothing
        return quote
            GlobalLogger[].to_console && println(stdout, $(esc(val...)))
            println(GlobalLogger[].io, $(esc(val...)))
            flush(GlobalLogger[].io)
        end
    end
end
