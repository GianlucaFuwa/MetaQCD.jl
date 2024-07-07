function autoc_time_int(x::Vector{<:Real})
    M = length(x)
    avg = mean(x)

    C = Γ₀ = zero(Float64) 

    @tturbo for k in 1:(M - 1)
        Γ₀ += (x[k] - avg) * (x[k + 1] - avg)
    end

    Γ₀ /= M - 1

    for t in 2:M-1
        tmp = 0.0

        @tturbo for k in 1:(M - t)
            tmp += (x[k] - avg) * (x[k + t] - avg)
        end
    
        tmp_n = tmp / (M - t)

        (tmp_n <= 0) && break

        C += tmp_n
    end

    C = Γ₀ + 2C
    τ = C / 2Γ₀
    return τ
end

# function autoc_time_int(x::Vector{<:Real}, ::Any)
#     id = rand(Int64)
#     x_uw = uwreal(x, "#$id#")
#     uwerr(x_uw)
#     @show ADerrors.window(x_uw, "#$id#")
#     return taui(x_uw, "#$id#")
# end
