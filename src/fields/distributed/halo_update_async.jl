# CPU and GPU fields share the exchange-plan implementation. CPU launches and
# stream dependencies are synchronous, so no unscheduled Task placeholders or
# background MPI tasks are needed.
finalize_halo_update!(::Nothing) = nothing
finalize_halo_update!(::Tuple{}) = nothing
