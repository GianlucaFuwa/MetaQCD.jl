module Logs

using ..Utils
using Dates

export __GlobalLogger, MetaLogger, current_time, @level1, @level2, @level3, @level4
export fclose, fopen, printf, prints_to_console, newline, set_global_logger!

@inline current_time() = Dates.now(UTC)

include("printf.jl")
include("verbose.jl")

end
