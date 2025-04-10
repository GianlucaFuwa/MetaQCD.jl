function temper!( # INFO: When using MPI in tempering
    U::Gaugefield,
    bias::Bias,
    numaccepts_temper,
    instance_state,
    swap_every,
    itrj;
    recalc=false,
)
    itrj%swap_every != 0 && return nothing
    recalc && recalc_cv!(U, bias)
    instance_comm = mpi_comm_instance()
    root_comm = mpi_comm_root()
    numinstances = MPI_NUMINSTANCES[]
    myrank = mpi_myrank(root_comm)
    mpi_barrier()
    
    # Query `instance_state` to find out which rank has to temper with which
    # Convention: instance N <-> instance N-1, instance N-1 <-> instance N-2, etc.
    for i in (numinstances-1):-1:1
        # Determine the ranks that have instances i and i-1
        rank_i = get_rank_from_instance(i, instance_state)
        rank_i_min_1 = get_rank_from_instance(i-1, instance_state)

        if (myrank == rank_i || myrank == rank_i_min_1) && mpi_amroot(instance_comm)
            if myrank == rank_i
                mpi_ssend(U.CV::Vector{Float64}, root_comm; dest=rank_i_min_1::Int64, tag=1) 
                CV_j = mpi_srecv(root_comm; source=rank_i_min_1::Int64, tag=1)
            elseif myrank == rank_i_min_1
                CV_j = mpi_srecv(root_comm; source=rank_i::Int64, tag=1)
                mpi_ssend(U.CV::Vector{Float64}, root_comm; dest=rank_i::Int64, tag=1) 
            end

            if myrank == rank_i
                ΔV1 = bias(CV_j) - bias(U.CV)
                ΔV2 = mpi_recv(Float64, root_comm; source=rank_i_min_1::Int64, tag=2)
                acc_prob = exp(-ΔV1 - ΔV2)
                is_accepted = rand() ≤ acc_prob
                mpi_send(is_accepted::Bool, root_comm; dest=rank_i_min_1::Int64, tag=3) 
            elseif myrank == rank_i_min_1 
                ΔV2 = bias(CV_j) - bias(U.CV)
                mpi_send(ΔV2::Float64, root_comm; dest=rank_i::Int64, tag=2) 
                is_accepted = mpi_recv(Bool, root_comm; source=rank_i::Int64, tag=3)
            end

            if is_accepted
                if myrank == rank_i
                    mpi_ssend(bias.bias, root_comm; dest=rank_i_min_1, tag=3)
                    new_bias = mpi_srecv(root_comm; source=rank_i_min_1, tag=3)

                    bias.bias = new_bias

                    instance_state[rank_i+1] = i-1
                    instance_state[rank_i_min_1+1] = i

                    # Update the local instance variable
                    MPI_INSTANCE[] = i-1
                    numaccepts_temper[i] += 1
                    @level1 "|  Old/New: $(i) -> $(i-1)"
                elseif myrank == rank_i_min_1
                    mpi_ssend(bias.bias, root_comm; dest=rank_i, tag=3)
                    new_bias = mpi_srecv(root_comm; source=rank_i, tag=3)

                    bias.bias = new_bias

                    instance_state[rank_i_min_1+1] = i
                    instance_state[rank_i+1] = i-1

                    # Update the local instance variable
                    MPI_INSTANCE[] = i
                    @level1 "|  Old/New: $(i-1) -> $(i)"
                end
            end

            # Synchronize the roots of each instance
            mpi_bcast!(instance_state, root_comm; root=rank_i)
            mpi_bcast!(numaccepts_temper, root_comm; root=rank_i)
        end

        mpi_barrier()

        # Synchronize ranks within instance
        mpi_bcast!(instance_state, instance_comm; root=0)
        mpi_bcast!(numaccepts_temper, instance_comm; root=0)

        bias.bias = mpi_bcast(bias.bias, instance_comm; root=0)

        acc_pct = 100numaccepts_temper[i] / (itrj/swap_every)
        @level1 "|    Acceptance [$i <-> $(i-1)]:\t$(acc_pct) %"
    end

    return nothing
end

function temper!( # INFO: When not using MPI in tempering
    U::Vector{TG}, bias::Vector{TB}, numaccepts_temper, swap_every, itrj; recalc=false
) where {TG<:Gaugefield,TB<:Bias}
    itrj % swap_every != 0 && return nothing
    numinstances = length(U)
    recalc && recalc_cv!(U[1], bias[1])

    for i in numinstances:-1:2
        U1 = U[i]
        U2 = U[i-1]
        bias1 = bias[i]
        bias2 = bias[i-1]
        cv1 = U1.CV
        cv2 = U2.CV
        ΔV1 = bias1(cv2) - bias1(cv1)
        ΔV2 = bias2(cv1) - bias2(cv2)
        acc_prob = exp(-ΔV1 - ΔV2)
        @level1("|  delta_V$(i) = $(ΔV1)\tdelta_V$(i-1) = $(ΔV2)")

        if rand() ≤ acc_prob
            @level1 "|  Swap accepted"
            numaccepts_temper[i-1] += 1
            swap_U!(U1, U2)
            update_bias!(bias1, cv2, itrj)
            update_bias!(bias2, cv1, itrj)
        else
            @level1 "|  Swap rejected"
        end

        acc_pct = 100numaccepts_temper[i-1] / (itrj/swap_every)
        @level1 "|    Acceptance [$i <-> $(i-1)]:\t$(acc_pct) %"
    end

    return nothing
end

function swap_U!(a, b)
    check_dims(a, b)
    a_Sg_tmp = deepcopy(a.Sg)
    a_CV_tmp = deepcopy(a.CV)

    a.Sg = b.Sg
    a.CV = b.CV
    b.Sg = a_Sg_tmp
    b.CV = a_CV_tmp

    @batch for site in eachindex(a)
        for μ in 1:4
            a_tmp = a[μ, site]
            a[μ, site] = b[μ, site]
            b[μ, site] = a_tmp
        end
    end

    return nothing
end

@inline function get_rank_from_instance(myinstance, state)
    rank = findfirst(x -> x == myinstance, state) - 1
    return rank
end
