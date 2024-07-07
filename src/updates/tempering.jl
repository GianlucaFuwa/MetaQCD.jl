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
    itrj % swap_every != 0 && return nothing
    recalc && recalc_CV!(U, bias)
    
    # Query `instance_state` to find out which rank has to temper with which
    # Convention: instance N <-> instance N-1, instance N-1 <-> instance N-2, etc.
    for i in (COMM_SIZE-1):-1:2
        # Determine the ranks that have instances i and i-1
        rank_i = get_rank_for_instance(i, instance_state)
        rank_i_minus_1 = get_rank_for_instance(i-1, instance_state)

        if MYRANK == rank_i || MYRANK == rank_i_minus_1
            if MYRANK == rank_i
                MPI.Send(U.CV::Float64, COMM; dest=rank_i_minus_1::Int64, tag=0) 
                CV_j = MPI.Recv(Float64, COMM; source=rank_i_minus_1::Int64, tag=0)
            else MYRANK == rank_i_minus_1
                CV_j = MPI.Recv(Float64, COMM; source=rank_i::Int64, tag=0)
                MPI.Send(U.CV::Float64, COMM; dest=rank_i::Int64, tag=0) 
            end

            if MYRANK == rank_i
                ΔV1 = bias(CV_j) - bias(U.CV)
                ΔV2 = MPI.Recv(Float64, COMM; source=rank_i_minus_1::Int64, tag=1)
                acc_prob = exp(-ΔV1 - ΔV2)
                is_accepted = rand() ≤ acc_prob
                MPI.Send(is_accepted::Bool, COMM; dest=rank_i_minus_1::Int64, tag=2) 
            else 
                ΔV2 = bias(CV_j) - bias(U.CV)
                MPI.Send(ΔV2::Float64, COMM; dest=rank_i::Int64, tag=1) 
                is_accepted = MPI.Recv(Bool, COMM; source=rank_i::Int64, tag=2)
            end

            if is_accepted
                if MYRANK == rank_i
                    instance_state[MYRANK+1] = i-1
                    instance_state[rank_i_minus_1+1] = i

                    # Update the local instance variable
                    myinstance[] = i-1
                elseif MYRANK == rank_i_minus_1
                    instance_state[MYRANK+1] = i
                    instance_state[rank_i+1] = i-1

                    # Update the local instance variable
                    myinstance[] = i
                    numaccepts_temper[i-1] += 1
                end
            end
        end

        # Synchronize vectors across all processes
        MPI.Bcast!(instance_state, COMM; root=rank_i)
        MPI.Bcast!(numaccepts_temper, COMM; root=rank_i)
        MPI.Barrier(COMM)
        @level1(
            "|  Acceptance [$i ⇔  $(i-1)]:\t$(100numaccepts_temper[i-1] / (itrj/swap_every)) %"
        )
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

        @level1(
            "|  Acceptance [$i ⇔  $(i-1)]:\t$(100numaccepts_temper[i-1] / (itrj/swap_every)) %"
        )
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

@inline get_rank_for_instance(myinstance, state) = findfirst(x -> x == myinstance, state)
