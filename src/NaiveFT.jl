struct naiveSubLatticeFT{N,Vec<:AbstractVector{<:Number},Mat<:NTuple{N,AbstractVector{<:Number}}} <: AbstractSubLatticeFT
    Sij::Vec
    Rij::Mat
end

@inline function (F::naiveSubLatticeFT)(kx::AbstractFloat, ky::AbstractFloat)
    (; Sij, Rij) = F
    Chi_k_im = zero(eltype(Sij))
    Chi_k_re = zero(eltype(Sij))
    Rx, Ry = Rij

    @turbo for i in eachindex(Sij, Rx, Ry)
        rx = Rx[i]
        ry = Ry[i]

        res_im, res_re = sincos(kx * rx + ky * ry)

        Chi_k_im += res_im * Sij[i]
        Chi_k_re += res_re * Sij[i]

    end

    return Chi_k_re + 1im * Chi_k_im
end

@inline function (F::naiveSubLatticeFT)(kx::AbstractFloat, ky::AbstractFloat, kz::AbstractFloat)
    (; Sij, Rij) = F
    Chi_k_im = zero(eltype(Sij))
    Chi_k_re = zero(eltype(Sij))
    Rx, Ry, Rz = Rij

    @turbo for i in eachindex(Sij, Rx, Ry)
        rx = Rx[i]
        ry = Ry[i]
        rz = Rz[i]

        res_im, res_re = sincos(kx * rx + ky * ry + kz * rz)

        Chi_k_im += res_im * Sij[i]
        Chi_k_re += res_re * Sij[i]

    end

    return Chi_k_re + 1im * Chi_k_im
end


@inline function (F::naiveSubLatticeFT)(kx::Real, ky::Real, kz::Real)
    (; Sij, Rij) = F
    Chi_k_im = zero(eltype(Sij))
    Chi_k_re = zero(eltype(Sij))
    Rx, Ry, Rz = Rij

    for i in eachindex(Sij, Rx, Ry)
        rx = Rx[i]
        ry = Ry[i]
        rz = Rz[i]

        res_im, res_re = sincos(kx * rx + ky * ry + kz * rz)

        Chi_k_im += res_im * Sij[i]
        Chi_k_re += res_re * Sij[i]

    end

    return Chi_k_re + 1im * Chi_k_im
end

@inline function (F::naiveSubLatticeFT)(kx::Real, ky::Real)
    (; Sij, Rij) = F
    Chi_k_im = zero(eltype(Sij))
    Chi_k_re = zero(eltype(Sij))
    Rx, Ry = Rij

    for i in eachindex(Sij, Rx, Ry)
        rx = Rx[i]
        ry = Ry[i]

        res_im, res_re = sincos(kx * rx + ky * ry)

        Chi_k_im += res_im * Sij[i]
        Chi_k_re += res_re * Sij[i]

    end

    return Chi_k_re + 1im * Chi_k_im
end

@inline function (F::naiveSubLatticeFT)(k::Vararg{Number,NArgs}) where {NArgs}
    (; Sij, Rij) = F
    Chi_k = zero(Complex{eltype(Sij)})

    @inbounds for i in eachindex(Sij)
        arg = sum(k[j] * Rij[j][i] for j in 1:NArgs)

        Chi_k += exp(1im * arg) * Sij[i]

    end
    return Chi_k
end

@inline function (A::naiveSubLatticeFT)(k::SVector)
    return A(k...)
end

function splitRijAndSij(
    S_ab::AbstractMatrix{<:AbstractArray},
    BasisVectors::AbstractMatrix,
    UnitCellVectors::AbstractArray{SVector{Dim, T}}
) where {Dim,T}

    NCell, NCell2 = size(S_ab)
    @assert NCell == NCell2 "S_ab needs to be a square matrix"
    Sij_vec = [reshape(S_ab[α, β], length(S_ab[α, β])) for α in 1:NCell, β in 1:NCell]

    a = BasisVectors
    UC = UnitCellVectors

    function Rij(α, β)::Vector{SVector{Dim, T}}
        dim = size(S_ab[α, β])
        L = dim[1] ÷ 2
        range = L:-1:-L
        latticeInds = Iterators.product((range for _ in 1:Dim)...)
        rij = [a * SVector(Tuple(n)) + (UC[α] - UC[β]) for n in latticeInds]
        return reshape(rij, length(rij))

    end

    Rij_vec = [Rij(α, β) for α in 1:NCell, β in 1:NCell]

    return (; Rij_vec, Sij_vec)

end

function splitRij(Rij::Vector{SVector{2,T}})::NTuple{2,Vector{T}} where {T<:Real}
    return (getindex.(Rij, 1), getindex.(Rij, 2))
end

function splitRij(Rij::Vector{SVector{3,T}})::NTuple{3,Vector{T}} where {T<:Real}
    return (getindex.(Rij, 1), getindex.(Rij, 2), getindex.(Rij, 3))
end

# function splitRij(Rij::Vector{SVector{Dim,T}}) where {T<:Real,Dim}
#     return Tuple(getindex.(Rij, i) for i in 1:Dim)
# end

function naiveLatticeFT(
    S_ab::AbstractMatrix{<:AbstractArray},
    BasisVectors::AbstractMatrix,
    UnitCellVectors::AbstractArray{<:AbstractArray}
)
    (; Rij_vec, Sij_vec) = splitRijAndSij(S_ab, BasisVectors, UnitCellVectors)
    Rij_vec2 = splitRij.(Rij_vec)
    return LatticeFT(naiveSubLatticeFT.(Sij_vec, Rij_vec2))
end
