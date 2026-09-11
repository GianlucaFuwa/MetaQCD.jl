using Random
using MetaQCD
using MetaQCD.Utils
using MetaQCD.Measurements: polyakov_traced, polyakov_deriv!, polyakov_mag_deriv!, polyakov_phase_deriv!
using Test
# using AMDGPU

function test_pderivative(
    ; backend=CPU,
    nprocs_cart=(1, 1, 1, 1),
    halo_width=1,
    N=4,
    GA=WilsonGaugeAction,
)
    Random.seed!(123)
    mpi_amroot() && println("Gauge and Clover derivative test")
    @assert N in (4, 16)

    relerrors = Matrix{Float64}(undef, 8, 8)

    @testset "Gauge derivative" begin
        NX = NY = NZ = NT = N
        Ucpu = Gaugefield{CPU,Float64,GA,12}(
            NX, NY, NZ, NT, 6.0, numprocs_cart=nprocs_cart, halo_width=halo_width
        )

        filename = if nprocs_cart != (1, 1, 1, 1)
            pkgdir(MetaQCD, "test", NX==4 ? "testconf_mpi" : "testconf_16_mpi")
        else
            pkgdir(MetaQCD, "test", NX==4 ? "testconf.txt" : "testconf_16.txt")
        end

        load_field!(BridgeFormat(), Ucpu, filename)

        if backend !== CPU
            U = MetaQCD.convert_field(backend, Ucpu)
        else
            U = Ucpu
        end

        # gaction_old = calc_gauge_action(U)
        # topcharge_old = top_charge(U)

        # Test for smearing with 5 steps and stout parameter 0.12
        smearing = StoutSmearing(U; numlayers=5, rho=0.12)

        staples = Colorfield(U)
        temp_force = Colorfield(U)
        temp_force2 = Colorfield(U)
        dLdU_re = Colorfield(U)
        dLdU_im = Colorfield(U)
        dLdU_abs = Colorfield(U)
        dLdU_phase = Colorfield(U)
        dLdU_resmeared = Colorfield(U)
        dLdU_imsmeared = Colorfield(U)
        dLdU_abssmeared = Colorfield(U)
        dLdU_phasesmeared = Colorfield(U)

        coord = (2, 3, 1, 2)
        site = SiteCoords(coord...)
        μ = 4
        ΔH = 0.00001

        for group_direction in 1:8
            # Unsmeared
            Ufwdcpu = deepcopy(Ucpu)
            if site in eachindex(Ucpu)
                Ufwdcpu[μ, site] = expλ(group_direction, ΔH) * Ufwdcpu[μ, site]
            end
            Ufwd = convert_field(backend, Ufwdcpu)
            ploop_new_fwd = polyakov_traced(Ufwd)
            ploop_re_new_fwd = real(ploop_new_fwd)
            ploop_im_new_fwd = imag(ploop_new_fwd)
            ploop_abs_new_fwd = abs(ploop_new_fwd)
            ploop_phase_new_fwd = angle(ploop_new_fwd)

            Ubwdcpu = deepcopy(Ucpu)
            if site in eachindex(Ucpu)
                Ubwdcpu[μ, site] = expλ(group_direction, -ΔH) * Ubwdcpu[μ, site]
            end
            Ubwd = convert_field(backend, Ubwdcpu)
            ploop_new_bwd = polyakov_traced(Ubwd)
            ploop_re_new_bwd = real(ploop_new_bwd)
            ploop_im_new_bwd = imag(ploop_new_bwd)
            ploop_abs_new_bwd = abs(ploop_new_bwd)
            ploop_phase_new_bwd = angle(ploop_new_bwd)

            # Smeared
            calc_smearedU!(smearing, Ufwd)
            ploop_new_fwd_smeared = polyakov_traced(smearing.Usmeared_multi[end])
            ploop_re_new_fwd_smeared = real(ploop_new_fwd_smeared)
            ploop_im_new_fwd_smeared = imag(ploop_new_fwd_smeared)
            ploop_abs_new_fwd_smeared = abs(ploop_new_fwd_smeared)
            ploop_phase_new_fwd_smeared = angle(ploop_new_fwd_smeared)

            calc_smearedU!(smearing, Ubwd)
            ploop_new_bwd_smeared = polyakov_traced(smearing.Usmeared_multi[end])
            ploop_re_new_bwd_smeared = real(ploop_new_bwd_smeared)
            ploop_im_new_bwd_smeared = imag(ploop_new_bwd_smeared)
            ploop_abs_new_bwd_smeared = abs(ploop_new_bwd_smeared)
            ploop_phase_new_bwd_smeared = angle(ploop_new_bwd_smeared)

            polyakov_deriv_bare!(dLdU_re, temp_force, U, nothing, NoSmearing(), 1)
            polyakov_deriv_bare!(dLdU_im, temp_force, U, nothing, NoSmearing(), -im)
            polyakov_mag_deriv_bare!(dLdU_abs, temp_force, U, nothing, NoSmearing())
            polyakov_phase_deriv_bare!(dLdU_phase, temp_force, U, nothing, NoSmearing())
            polyakov_deriv_bare!(dLdU_resmeared, temp_force2, U, temp_force, smearing, 1)
            polyakov_deriv_bare!(dLdU_imsmeared, temp_force2, U, temp_force, smearing, -im)
            polyakov_mag_deriv_bare!(dLdU_abssmeared, temp_force2, U, temp_force, smearing)
            polyakov_phase_deriv_bare!(dLdU_phasesmeared, temp_force2, U, temp_force, smearing)

            if site in eachindex(U)
                dploop_re_proj = real(multr(im * λ[group_direction], dLdU_re[μ, site]))
                dploop_im_proj = real(multr(im * λ[group_direction], dLdU_im[μ, site]))
                dploop_abs_proj = real(multr(im * λ[group_direction], dLdU_abs[μ, site]))
                dploop_phase_proj = real(multr(im * λ[group_direction], dLdU_phase[μ, site]))
                dploop_re_proj_smeared = real(multr(im * λ[group_direction], dLdU_resmeared[μ, site]))
                dploop_im_proj_smeared = real(multr(im * λ[group_direction], dLdU_imsmeared[μ, site]))
                dploop_abs_proj_smeared = real(multr(im * λ[group_direction], dLdU_abssmeared[μ, site]))
                dploop_phase_proj_smeared = real(multr(im * λ[group_direction], dLdU_phasesmeared[μ, site]))
            else
                dploop_re_proj = 1.0
                dploop_im_proj = 1.0
                dploop_abs_proj = 1.0
                dploop_phase_proj = 1.0
                dploop_re_proj_smeared = 1.0
                dploop_im_proj_smeared = 1.0
                dploop_abs_proj_smeared = 1.0
                dploop_phase_proj_smeared = 1.0
            end

            # if group_direction == 1
            #     @show ploop_re_new_fwd
            # end

            pl_re_symm_diff = (ploop_re_new_fwd - ploop_re_new_bwd) / 2ΔH
            pl_im_symm_diff = (ploop_im_new_fwd - ploop_im_new_bwd) / 2ΔH
            pl_abs_symm_diff = (ploop_abs_new_fwd - ploop_abs_new_bwd) / 2ΔH
            pl_phase_symm_diff = (ploop_phase_new_fwd - ploop_phase_new_bwd) / 2ΔH
            pl_re_symm_diff_smeared = (ploop_re_new_fwd_smeared - ploop_re_new_bwd_smeared) / 2ΔH
            pl_im_symm_diff_smeared = (ploop_im_new_fwd_smeared - ploop_im_new_bwd_smeared) / 2ΔH
            pl_abs_symm_diff_smeared = (ploop_abs_new_fwd_smeared - ploop_abs_new_bwd_smeared) / 2ΔH
            pl_phase_symm_diff_smeared = (ploop_phase_new_fwd_smeared - ploop_phase_new_bwd_smeared) / 2ΔH

            relerrors[group_direction, 1] = (pl_re_symm_diff - dploop_re_proj) / pl_re_symm_diff
            relerrors[group_direction, 2] = (pl_im_symm_diff - dploop_im_proj) / pl_im_symm_diff
            relerrors[group_direction, 3] = (pl_abs_symm_diff - dploop_abs_proj) / pl_abs_symm_diff
            relerrors[group_direction, 4] = (pl_phase_symm_diff - dploop_phase_proj) / pl_phase_symm_diff
            relerrors[group_direction, 5] = (pl_re_symm_diff_smeared - dploop_re_proj_smeared) / pl_re_symm_diff_smeared
            relerrors[group_direction, 6] = (pl_im_symm_diff_smeared - dploop_im_proj_smeared) / pl_im_symm_diff_smeared
            relerrors[group_direction, 7] = (pl_abs_symm_diff_smeared - dploop_abs_proj_smeared) / pl_abs_symm_diff_smeared
            relerrors[group_direction, 8] = (pl_phase_symm_diff_smeared - dploop_phase_proj_smeared) / pl_phase_symm_diff_smeared

            if mpi_amroot()
                println("================= Group direction $(group_direction) =================")
                println(
                    "/ PL re rel. error (unsmeared): \t", (pl_re_symm_diff - dploop_re_proj) / pl_re_symm_diff,
                )
                println(
                    "/ PL im rel. error (unsmeared): \t", (pl_im_symm_diff - dploop_im_proj) / pl_im_symm_diff,
                )
                println(
                    "/ PL abs rel. error (unsmeared): \t", (pl_abs_symm_diff - dploop_abs_proj) / pl_abs_symm_diff,
                )
                println(
                    "/ PL phase rel. error (unsmeared): \t", (pl_phase_symm_diff - dploop_phase_proj) / pl_phase_symm_diff,
                )
                println(
                    "/ PL re rel. error (smeared):   \t", (pl_re_symm_diff_smeared - dploop_re_proj_smeared) / pl_re_symm_diff_smeared,
                )
                println(
                    "/ PL im rel. error (smeared):   \t", (pl_im_symm_diff_smeared - dploop_im_proj_smeared) / pl_im_symm_diff_smeared,
                )
                println(
                    "/ PL abs rel. error (smeared):   \t", (pl_abs_symm_diff_smeared - dploop_abs_proj_smeared) / pl_abs_symm_diff_smeared,
                )
                println(
                    "/ PL phase rel. error (smeared):   \t", (pl_phase_symm_diff_smeared - dploop_phase_proj_smeared) / pl_phase_symm_diff_smeared,
                )
                println("")
            end
        end

        if mpi_amroot()
            println()

            @testset "Polyakov loop derivative" begin
                @test sum(relerrors[:, 2]) / length(relerrors[:, 2]) < 1e-4
            end
        end
    end

    mpi_barrier()
    return relerrors
end

function polyakov_deriv_bare!(dU, temp_force2, U, temp_force, smearing, deriv_fac)
    if isnothing(temp_force)
        polyakov_deriv!(dU, U, deriv_fac)
    else
        calc_smearedU!(smearing, U)
        fully_smeared_U = smearing.Usmeared_multi[end]
        polyakov_deriv!(dU, fully_smeared_U, deriv_fac)
        stout_backprop!(dU, temp_force, smearing)
    end

    return nothing
end

function polyakov_mag_deriv_bare!(dU, temp_force2, U, temp_force, smearing)
    if isnothing(temp_force)
        polyakov_mag_deriv!(dU, temp_force2, U)
    else
        calc_smearedU!(smearing, U)
        fully_smeared_U = smearing.Usmeared_multi[end]
        polyakov_mag_deriv!(dU, temp_force2, fully_smeared_U)
        stout_backprop!(dU, temp_force, smearing)
    end

    return nothing
end

function polyakov_phase_deriv_bare!(dU, temp_force2, U, temp_force, smearing)
    if isnothing(temp_force)
        polyakov_phase_deriv!(dU, temp_force2, U)
    else
        calc_smearedU!(smearing, U)
        fully_smeared_U = smearing.Usmeared_multi[end]
        polyakov_phase_deriv!(dU, temp_force2, fully_smeared_U)
        stout_backprop!(dU, temp_force, smearing)
    end

    return nothing
end

# AMDGPU.@allowscalar test_pderivative(; backend=ROCBackend, nprocs_cart=(1, 1, 1, mpi_size()))
