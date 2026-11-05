using MetaQCD
using Test

function test_rhmc()
    R = MetaQCD.RHMCParameters
    p = R.RHMCParams(1//8; n_max=10, lambda_low=0.001, lambda_high=4.0, tol=1e-5)
    @testset "RHMC rational coefficients" begin
        # @test R.get_n(p) == R.get_n_inverse(p) == 5
        @test maximum(p.maxerr) <= 1e-5
        f = R.fittedfunction(p.coeffs)
        g = R.fittedfunction(p.coeffs_inverse)
        xs = exp.(range(log(0.1), log(4.0); length=10001))
        @test maximum(abs.(f.(xs) .* g.(xs) .- 1)) < 1e-12
        @test maximum(abs.(f.(xs) .- sqrt.(xs))) <= 1e-5
        @test maximum(abs.(g.(xs) .- inv.(sqrt.(xs)))) <= 1e-5
        @test all(>(0), p.coeffs.β)
        @test all(>(0), p.coeffs_inverse.β)
        # Every lower order is checked after refinement before accepting order 5.
        capped = @test_logs (:warn, r"RHMC approximation did not meet tolerance") match_mode=:any R.RHMCParams(
            1//2; n_max=4, lambda_low=0.1, lambda_high=4.0, tol=1e-5)
        @test R.get_n(capped) <= 4
        @test maximum(capped.maxerr) > 1e-5
        # A loose tolerance must stop on the very first candidate.
        loose = R.RHMCParams(1//2; n_max=8, lambda_low=0.1, lambda_high=4.0, tol=1.0)
        @test R.get_n(loose) == 1
    end

    return p
end

abspath(PROGRAM_FILE) == (@__FILE__) && test_rhmc()
