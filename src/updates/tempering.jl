function temper!(
    U::Gaugefield,
    bias::Bias,
    numaccepts_temper,
    instance_state,
    myinstance,
    swap_every,
    itrj;
    recalc=false,
)
    itrj%swap_every != 0 && return nothing
    recalc && recalc_CV!(U, bias)
    comm = mpi_comm()
    myrank = mpi_myrank()
    mpi_barrier()
    
    # Query `instance_state` to find out which rank has to temper with which
    # Convention: instance N <-> instance N-1, instance N-1 <-> instance N-2, etc.
    for i in (mpi_size()-1):-1:1
        # Determine the ranks that have instances i and i-1
        rank_i = get_rank_from_instance(i, instance_state)
        rank_i_minus_1 = get_rank_from_instance(i-1, instance_state)

        if myrank == rank_i || myrank == rank_i_minus_1
            if myrank == rank_i
                mpi_send(U.CV::Float64, comm; dest=rank_i_minus_1::Int64, tag=0) 
                CV_j = mpi_recv(Float64, comm; source=rank_i_minus_1::Int64, tag=0)
            else myrank == rank_i_minus_1
                CV_j = mpi_recv(Float64, comm; source=rank_i::Int64, tag=0)
                mpi_send(U.CV::Float64, comm; dest=rank_i::Int64, tag=0) 
            end

            if myrank == rank_i
                ΔV1 = bias(CV_j) - bias(U.CV)
                ΔV2 = mpi_recv(Float64, comm; source=rank_i_minus_1::Int64, tag=1)
                acc_prob = exp(-ΔV1 - ΔV2)
                is_accepted = rand() ≤ acc_prob
                mpi_send(is_accepted::Bool, comm; dest=rank_i_minus_1::Int64, tag=2) 
            else 
                ΔV2 = bias(CV_j) - bias(U.CV)
                mpi_send(ΔV2::Float64, comm; dest=rank_i::Int64, tag=1) 
                is_accepted = mpi_recv(Bool, comm; source=rank_i::Int64, tag=2)
            end

            if is_accepted
                if myrank == rank_i
                    instance_state[myrank+1] = i-1
                    instance_state[rank_i_minus_1+1] = i

                    # Update the local instance variable
                    myinstance[] = i-1
                elseif myrank == rank_i_minus_1
                    instance_state[myrank+1] = i
                    instance_state[rank_i+1] = i-1

                    # Update the local instance variable
                    myinstance[] = i
                    numaccepts_temper[i-1] += 1
                end
            end
        end

        # Synchronize vectors across all processes
        mpi_bcast!(instance_state, comm; root=rank_i)
        mpi_bcast!(numaccepts_temper, comm; root=rank_i)
        mpi_barrier()
        @level1 """
        |  Acceptance [$i ⇔  $(i-1)]:\t$(100numaccepts_temper[i-1] / (itrj/swap_every)) %
        """
    end
    return nothing
end

function temper!(
    U::Vector{TG}, bias::Vector{TB}, numaccepts_temper, swap_every, itrj; recalc=false
) where {TG<:Gaugefield,TB<:Bias}
    itrj % swap_every != 0 && return nothing
    numinstances = length(U)
    recalc && recalc_CV!(U[1], bias[1])

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
        @level1("|  ΔV$(i) = $(ΔV1)\tΔV$(i-1) = $(ΔV2)")

        if rand() ≤ acc_prob
            println("# swap accepted")
            numaccepts_temper[i-1] += 1
            swap_U!(U1, U2)
            update_bias!(bias1, cv2, itrj, true)
            update_bias!(bias2, cv1, itrj, true)
        else
            println("# swap rejected")
        end

        @level1 """
        "|  Acceptance [$i ⇔  $(i-1)]:\t$(100numaccepts_temper[i-1] / (itrj/swap_every)) %
        """
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

    update_halo!(a)
    update_halo!(b)
    return nothing
end

@inline function get_rank_from_instance(myinstance, state)
    rank = findfirst(x -> x == myinstance, state) - 1
    return rank
end
