# From https://github.com/akio-tomiya/LatticeDiracOperators.jl/blob/master/src/rhmc/AlgRemez.jl
"""
    AlgRemez.jl is a wrapper for AlgRemez written in c++. 
    Please see
    https://github.com/maddyscientist/AlgRemez
"""
module AlgRemez

using AlgRemez_jll

struct AlgRemezCoeffs{N}
    α0::Float64
    α::NTuple{N,Float64}
    β::NTuple{N,Float64}
    n::Int64
end

function Base.display(x::AlgRemezCoeffs)
    println("""
        f(x) = α0 + sum_i^n α[i]/(x + β[i])
    """)
    println("Order: $(x.n)")
    println("α0: $(x.α0)")
    println("α: $(x.α)")
    return println("β: $(x.β)")
end

function fittedfunction(coeff::AlgRemezCoeffs)
    function func(x)
        value = coeff.α0
        for i in 1:coeff.n
            value += coeff.α[i] / (x + coeff.β[i])
        end
        return value
    end
    return x -> func(x)
end

"""
    calc_coefficients(y, z, n, lambda_low, lambda_high; precision=42)
    
Calculate the Remez coefficients of the function `f(x) = x^(-y/z)` up to order `n`
over the spectral range `[lambda_low, lambda_high]`, using precision digits of precision
in the arithmetic. \\ 
The parameters `y` and `z` must be positive, the approximation to f(x) = x^(-y/z) is simply
the inverse of the approximation to f(x) = x^(y/z).
"""
function calc_coefficients(y::Int, z::Int, n::Int, lambda_low, lambda_high; precision=42)
    @assert y > 0 && z > 0 "Inputs y and z need to be positive"
    # Save and reuse already calcuatated approximations
    dirname = pwd() * "/src/rhmc/"
    filename = "approx_$(y)_$(z)_$(n)_$(lambda_low)_$(lambda_high)_$precision.dat"
    filename_err = "error_$(y)_$(z)_$(n)_$(lambda_low)_$(lambda_high)_$precision.dat"
    pathname = dirname * filename
    if !(filename in readdir(dirname))
        try
            run(`$(algremez()) $y $z $n $n $lambda_low $lambda_high $precision`)
        catch _
            error("""
                  Running algremez.exe failed. The most likely reason is that the required
                  shared libraries cannot be found / aren't on the PATH or LD_LIBRARY_PATH
                  """)
        end
        mv("approx.dat", pathname)
        mv("error.dat", dirname * filename_err)
    end
    datas = readlines(pathname)
    icount = 3
    αplus0 = parse(Float64, split(datas[icount], "=")[2])
    αplus = zeros(Float64, n)
    βplus = zeros(Float64, n)

    for i in 1:n
        icount += 1
        u = split(datas[icount], ",")
        αplus[i] = parse(Float64, split(u[1], "=")[2])
        βplus[i] = parse(Float64, split(u[2], "=")[2])
    end

    αplus_tup = tuple(αplus...)
    βplus_tup = tuple(βplus...)
    coeff_plus = AlgRemezCoeffs(αplus0, αplus_tup, βplus_tup, n) # x^(y/z)
    icount += 4
    αminus0 = parse(Float64, split(datas[icount], "=")[2])
    αminus = zeros(Float64, n)
    βminus = zeros(Float64, n)

    for i in 1:n
        icount += 1
        u = split(datas[icount], ",")
        αminus[i] = parse(Float64, split(u[1], "=")[2])
        βminus[i] = parse(Float64, split(u[2], "=")[2])
    end

    αminus_tup = tuple(αminus...)
    βminus_tup = tuple(βminus...)
    coeff_minus = AlgRemezCoeffs(αminus0, αminus_tup, βminus_tup, n) # x^(-y/z)
    return coeff_plus, coeff_minus
end

end
