using MetaQCD
using MetaQCD.Utils
using LinearAlgebra
using Random
using Test

f_eo = even_odd(Spinorfield{CPU,Float64,3}(4, 4, 4, 4))

function testfun(f_eo::SpinorfieldEO{B}) where {B}
    f = f_eo.parent
    update_halo!(f)
    out = MetaQCD.Fields.parallelfor_sum(eachindex(true, f), B) do site
    end
    return out
end
