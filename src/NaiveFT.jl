struct naiveSubLatticeFT{N,Vec<:AbstractVector{<:Number},Mat<:NTuple{N,AbstractVector{<:Number}}} <: AbstractSubLatticeFT
    Sij::Vec
    Rij::Mat
end

@inline function (F::naiveSubLatticeFT)(kx::Real, ky::Real)
    Chi_k_re = 0
    Chi_k_im = 0
    (; Sij, Rij) = F
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

@inline function (F::naiveSubLatticeFT)(kx::Real, ky::Real, kz::Real)
    Chi_k_re = 0
    Chi_k_im = 0
    (; Sij, Rij) = F
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

@inline function (F::naiveSubLatticeFT)(k::Vararg{Number,NArgs}) where {NArgs}
    Chi_k = 0
    (; Sij, Rij) = F

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
    UnitCellVectors::AbstractArray{<:AbstractArray}
)

    NCell, NCell2 = size(S_ab)
    @assert NCell == NCell2 "S_ab needs to be a square matrix"
    Sij_vec = [reshape(S_ab[α, β], length(S_ab[α, β])) for α in 1:NCell, β in 1:NCell]

    a = BasisVectors
    UC = UnitCellVectors

    function Rij(α, β)
        dim = size(S_ab[α, β])
        L = dim[1] ÷ 2
        range = L:-1:-L
        latticeInds = Iterators.product((range for _ in 1:length(dim))...)
        rij = [a * SVector(Tuple(n)) + (UC[α] - UC[β]) for n in latticeInds]
        return reshape(rij, length(rij))

    end

    Rij_vec = [Rij(α, β) for α in 1:NCell, β in 1:NCell]

    return (; Rij_vec, Sij_vec)

end

function splitRij(Rij::Vector{SVector{Dim,T}}) where {T<:Real,Dim}
    return Tuple(getindex.(Rij, i) for i in 1:Dim)
end

function naiveLatticeFT(
    S_ab::AbstractMatrix{<:AbstractArray},
    BasisVectors::AbstractMatrix,
    UnitCellVectors::AbstractArray{<:AbstractArray}
)
    (; Rij_vec, Sij_vec) = splitRijAndSij(S_ab, BasisVectors, UnitCellVectors)
    Rij_vec = splitRij.(Rij_vec)
    return LatticeFT(naiveSubLatticeFT.(Sij_vec, Rij_vec))
end