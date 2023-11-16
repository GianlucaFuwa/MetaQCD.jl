import InteractiveUtils

abstract type VerboseLevel end

struct Verbose1 <: VerboseLevel
    fp::Union{Nothing, IOStream}
    Verbose1() = new(nothing)
    Verbose1(::Nothing) = new(nothing)
    Verbose1(filename::String) = new(open(filename, "w"))
    Verbose1(fp::IOStream) = new(fp)
end

struct Verbose2 <: VerboseLevel
    fp::Union{Nothing, IOStream}
    Verbose2() = new(nothing)
    Verbose2(::Nothing) = new(nothing)
    Verbose2(filename::String) = new(open(filename, "w"))
    Verbose2(fp::IOStream) = new(fp)
end

struct Verbose3 <: VerboseLevel
    fp::Union{Nothing, IOStream}
    Verbose3() = new(nothing)
    Verbose3(::Nothing) = new(nothing)
    Verbose3(filename::String) = new(open(filename, "w"))
    Verbose3(fp::IOStream) = new(fp)
end

function Base.flush(v::VerboseLevel)
    v.fp≢nothing && flush(v.fp)
end

function Base.close(v::VerboseLevel)
    v.fp≢nothing && close(v.fp)
end

function InteractiveUtils.versioninfo(v::VerboseLevel)
    InteractiveUtils.versioninfo()
    v.fp≢nothing && InteractiveUtils.versioninfo(v.fp)
end

function println_verbose1(v::Verbose3, val...)
    println(val...)

    if v.fp ≢ nothing
        println(v.fp, val...)
    end

    return nothing
end

function println_verbose2(v::Verbose3, val...)
    println(val...)

    if v.fp ≢ nothing
        println(v.fp, val...)
    end

    return nothing
end

function println_verbose3(v::Verbose3, val...)
    println(val...)

    if v.fp ≢ nothing
        println(v.fp, val...)
    end

    return nothing
end

function println_verbose1(v::Verbose2, val...)
    println(val...)

    if v.fp ≢ nothing
        println(v.fp, val...)
    end

    return nothing
end

function println_verbose2(v::Verbose2, val...)
    println(val...)

    if v.fp ≢ nothing
        println(v.fp, val...)
    end

    return nothing
end

function println_verbose1(v::Verbose1, val...)
    println(val...)

    if v.fp ≢ nothing
        println(v.fp, val...)
    end

    return nothing
end

function print_verbose1(v::Verbose3, val...)
    print(val...)

    if v.fp ≢ nothing
        print(v.fp, val...)
    end

    return nothing
end

function print_verbose2(v::Verbose3, val...)
    print(val...)

    if v.fp ≢ nothing
        print(v.fp, val...)
    end

    return nothing
end

function print_verbose3(v::Verbose3, val...)
    print(val...)

    if v.fp ≢ nothing
        print(v.fp, val...)
    end

    return nothing
end

function print_verbose1(v::Verbose2, val...)
    print(val...)

    if v.fp ≢ nothing
        print(v.fp, val...)
    end

    return nothing
end

function print_verbose2(v::Verbose2, val...)
    print(val...)

    if v.fp ≢ nothing
        print(v.fp, val...)
    end

    return nothing
end

function print_verbose1(v::Verbose1, val...)
    print(val...)

    if v.fp ≢ nothing
        print(v.fp, val...)
    end

    return nothing
end

println_verbose1(::Nothing, val...) = nothing
println_verbose2(::Nothing, val...) = nothing
println_verbose3(::Nothing, val...) = nothing
println_verbose2(::Verbose1, val...) = nothing
println_verbose3(::Verbose1, val...) = nothing
println_verbose3(::Verbose2, val...) = nothing

print_verbose1(::Nothing, val...) = nothing
print_verbose2(::Nothing, val...) = nothing
print_verbose3(::Nothing, val...) = nothing
print_verbose2(::Verbose1, val...) = nothing
print_verbose3(::Verbose1, val...) = nothing
print_verbose3(::Verbose2, val...) = nothing
