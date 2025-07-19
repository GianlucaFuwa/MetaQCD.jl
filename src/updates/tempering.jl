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
    comm_instance = mpi_comm_instance()
    comm_shared = mpi_comm_shared()
    numinstances = MPI_NUMINSTANCES[]
    myrank = mpi_myrank(comm_shared)
    mpi_barrier()
    
    # Query `instance_state` to find out which rank has to temper with which
    # Convention: instance N <-> instance N-1, instance N-1 <-> instance N-2, etc.
    for i in (numinstances-1):-1:1
        # Determine the ranks that have instances i and i-1
        rank_i = get_rank_from_instance(i, instance_state)
        rank_i_min_1 = get_rank_from_instance(i-1, instance_state)

        if (myrank == rank_i || myrank == rank_i_min_1) && mpi_amroot(comm_instance)
            if myrank == rank_i
                mpi_send(mpi_buffer(bias.CV), comm_shared; dest=rank_i_min_1::Int64, tag=1) 
                CV_j = mpi_recv(comm_shared; source=rank_i_min_1::Int64, tag=1)
            elseif myrank == rank_i_min_1
                CV_j = mpi_recv(comm_shared; source=rank_i::Int64, tag=1)
                mpi_send(mpi_buffer(bias.CV), comm_shared; dest=rank_i::Int64, tag=1) 
            end

            if myrank == rank_i
                ΔV1 = bias(CV_j) - bias(bias.CV)
                ΔV2 = mpi_recv(Float64, comm_shared; source=rank_i_min_1::Int64, tag=2)
                acc_prob = exp(-ΔV1 - ΔV2)
                is_accepted = rand() ≤ acc_prob
                mpi_send(is_accepted::Bool, comm_shared; dest=rank_i_min_1::Int64, tag=3) 
            elseif myrank == rank_i_min_1 
                ΔV2 = bias(CV_j) - bias(bias.CV)
                mpi_send(ΔV2::Float64, comm_shared; dest=rank_i::Int64, tag=2) 
                is_accepted = mpi_recv(Bool, comm_shared; source=rank_i::Int64, tag=3)
            end

            if is_accepted
                if myrank == rank_i
                    for icv in eachindex(bias.CV)
                        buf = bias.buffers[icv]
                        pack_buffer!(buf, bias.bias[icv])
                        mpi_send(mpi_buffer(buf), comm_shared; dest=rank_i_min_1, tag=100+icv)
                        mpi_recv!(buf, comm_shared; source=rank_i_min_1, tag=100+icv)
                        unpack_buffer!(bias.bias[icv], buf)
                    end

                    instance_state[rank_i+1] = i-1
                    instance_state[rank_i_min_1+1] = i

                    # Update the local instance variable
                    MPI_INSTANCE[] = i-1
                    numaccepts_temper[i] += 1
                    @level1 "|  Old/New: $(i) -> $(i-1)"
                elseif myrank == rank_i_min_1
                    for icv in eachindex(bias.CV)
                        buf = bias.buffers[icv]
                        pack_buffer!(buf, bias.bias[icv])
                        mpi_send(mpi_buffer(buf), comm_shared; dest=rank_i, tag=100+icv)
                        mpi_recv!(buf, comm_shared; source=rank_i, tag=100+icv)
                        unpack_buffer!(bias.bias[icv], buf)
                    end

                    instance_state[rank_i_min_1+1] = i
                    instance_state[rank_i+1] = i-1

                    # Update the local instance variable
                    MPI_INSTANCE[] = i
                    @level1 "|  Old/New: $(i-1) -> $(i)"
                end
            end

            # Synchronize between instances
            mpi_bcast!(instance_state, comm_shared; root=rank_i)
            mpi_bcast!(numaccepts_temper, comm_shared; root=rank_i)
        end

        mpi_barrier()

        # Synchronize within instance XXX: not needed?
        mpi_bcast!(instance_state, comm_instance; root=0)
        mpi_bcast!(numaccepts_temper, comm_instance; root=0)

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
        cv1 = bias1.CV
        cv2 = bias2.CV
        ΔV1 = bias1(cv2) - bias1(cv1)
        ΔV2 = bias2(cv1) - bias2(cv2)
        acc_prob = exp(-ΔV1 - ΔV2)
        @level1("|  delta_V$(i) = $(ΔV1)\tdelta_V$(i-1) = $(ΔV2)")

        if rand() ≤ acc_prob
            @level1 "|  Swap accepted"
            numaccepts_temper[i-1] += 1
            swap_U!(U1, U2, bias1, bias2)
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

function swap_U!(a::TF, b::TF, biasa, biasb) where {B,T,M,TF<:Gaugefield{B,T,M}}
    a_CV_tmp = deepcopy(biasa.CV)
    biasa.CV = biasb.CV
    biasb.CV = a_CV_tmp

    parallelfor(allindices(a, b), B, Val(M), (), (a, b), (a, b)) do μsite, a, b
        a_tmp = a[μsite]
        a[μsite] = b[μsite]
        b[μsite] = a_tmp
    end

    return nothing
end

@inline function get_rank_from_instance(myinstance, state)
    rank = findfirst(x -> x == myinstance, state) - 1
    return rank
end
