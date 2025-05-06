Dict(
    "L" => NTuple{4,UInt64},
    "β" => Real,
    "gauge_action" => ("wilson", "symanzik_tree", "iwasaki", "dbw2"),
    "NC" => (3,),
    "numtherm" => UInt64,
    "numsteps" => UInt64,
    "initial" => ("cold", "hot"),
    "numprocs_cart" => NTuple{4,UInt64},
    "halo_width" => UInt64,
    "fermion_action" => ("staggered", "staggered_eo", "wilson", "wilson_eo"),
    "boundary_condition" => ("antiperiodic", "periodic"),
)
