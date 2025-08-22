@inline function sarray_to_vecs(v::SVector{3,Complex{T}}) where {T}
    Base.Cartesian.@nexprs 3 i -> (
        vec_i = SIMD.Vec{2,T}(v[i].re, v[i].im)
    )
    return vec_1, vec_2, vec_3
end

@inline function sarray_to_vecs(v::SVector{12,Complex{T}}) where {T}
    Base.Cartesian.@nexprs 6 i -> (
        vec_i = SIMD.Vec{4,T}(
            v[2(i-1)+1].re, v[2(i-1)+1].im, v[2(i-1)+2].re, v[2(i-1)+2].im
        );
    )
    return vec_1, vec_2, vec_3, vec_4, vec_5, vec_6
end

@inline function sarray_to_vecs(::Val{12}, v::SMatrix{3,3,Complex{T},9}) where {T}
    Base.Cartesian.@nexprs 3 i -> (
        vec_i = SIMD.Vec{4,T}(
            v[2(i-1)+1].re, v[2(i-1)+1].im, v[2(i-1)+2].re, v[2(i-1)+2].im
        )
    )
    return vec_1, vec_2, vec_3
end

@inline function sarray_to_vecs(::Val{18}, v::SMatrix{3,3,Complex{T},9}) where {T}
    vecs = ntuple(Val(9)) do i
        vec_i = SIMD.Vec{2,T}(v[i].re, v[i].im);
    end

    return vecs
end
