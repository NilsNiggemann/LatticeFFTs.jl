abstract type AbstractPadding end
struct AutomaticPadding <: AbstractPadding end

"""
Given a function χ(r) on a Bravais lattice, return a Fourier transform via FFT as a view.
Assumptions:
    - χ(r) is given in lattice coordinates, i.e. Chi[n1,n2,n3] corresponds to chi(0,n1*a1+n2*a2+n3*a3) 
    - The central index corresponds to chi(0,0). 
"""
function getFFT(ChiR, P=plan_fft(ChiR))
    chik = FFTView(P * ifftshift(ChiR))
    return chik
end

"""
Pads the susceptibility to the final size 'newDims'. If any of the new dimensions is less than the original one, the old size is maintained
"""
function padSusc(ChiR::AbstractArray{T}, newDims::Tuple) where {T<:Number}
    dims = size(ChiR)
    newDims = max.(dims, newDims)

    Origin = (newDims .- dims) .÷ 2 .+ 2
    return PaddedView(zero(T), ChiR, newDims, Origin)
end

"""
Pads the susceptibility to the nearest power of 2 (minimum 64). This should ensure a fast computation and a smooth interpolation.
"""
function padSusc(ChiR::AbstractArray{T}, ::AutomaticPadding) where {T<:Number}
    dims = size(ChiR)
    Nearest2Power = Tuple(ceil(Int, 2^log2(d)) for d in dims)
    minSize = 64
    newDims = Tuple(max(n, minSize) for n in Nearest2Power)
    return padSusc(ChiR, newDims)
end

function padSusc(ChiR::AbstractArray{T}, val::Integer) where {T<:Number}
    val <= 0 && return ChiR
    padSusc(ChiR, Tuple(val for s in size(ChiR)))
end


function getInterpolatedFFT(
    Chi_ij::AbstractArray{<:Real}, 
    padding=AutomaticPadding(), 
    args...;
    Interpolation = BSpline(Cubic())
    )
    Chi_ij = padSusc(Chi_ij, padding)
    dims = size(Chi_ij)
    nk = Tuple(0:N for N in dims)
    FFT = getFFT(Chi_ij, args...)[nk...]
    # chik = Interpolations.interpolate!(FFT,BSpline(Cubic(InPlace(OnGrid()))))
    chik = Interpolations.interpolate(FFT, Interpolation)
    chik = scale(chik, 2π ./ dims .* nk)
    chik = extrapolate(chik, Periodic())
    return chik
end

struct PhaseShiftedFFT{InterpolationType,BasisMat<:AbstractMatrix,PhaseVecType<:AbstractVector} <: AbstractSubLatticeFT
    S::InterpolationType
    T::BasisMat
    # UC::UCType
    PhaseVector::PhaseVecType
end

@inline function (F::PhaseShiftedFFT)(k::SVector{N}) where {N}
    exp(1im * k' * F.PhaseVector) * F.S((F.T' * k)...)
end

@inline function (F::PhaseShiftedFFT)(k::AbstractVector)
    exp(1im * k' * F.PhaseVector) * F.S((F.T' * k)...)
end

@inline function (F::PhaseShiftedFFT)(x::Vararg{Number,NArgs}) where {NArgs}
    k = SVector(x)
    return F(k)
end

struct LatticeFT{Mat<:AbstractMatrix{<:AbstractSubLatticeFT}} <: AbstractLatticeFT{AbstractSubLatticeFT}
    S::Mat
    function LatticeFT(S::Mat) where {Mat<:AbstractMatrix{<:AbstractSubLatticeFT}}
        @assert size(S, 1) == size(S, 2) "All elements of LatticeFFT need to have the same size"
        return new{Mat}(S)
    end
end

Base.size(S::LatticeFT) = size(S.S)
Base.getindex(S::LatticeFT, I::Vararg{Int, N}) where N	= getindex(S.S, I...)
Base.setindex!(S::LatticeFT, x, I::Vararg{Int, N}) where N = setindex!(S.S, x, I...)

Base.copy(S::LatticeFT) = LatticeFT(copy(S.S))

""" 
evaluate the full Fourier transform averaging over all sublattices. Semantically equivalent to
    ```
    function (A::LatticeFFT)(k::AbstractVector)
        dim = size(A.S)
        return real(sum(a(k) for a in diag(A)) + )  / dim[1]
    end
    ```
"""
function (A::AbstractLatticeFT)(k::AbstractVector)
    dim = size(A)#
    res = 0
    for i in axes(A, 1)
        for j in axes(A, 2)
            if i == j
                res += real(A[i, j])(k)
            elseif i < j
                res += 2real(A[i, j])(k)
            end
        end
    end
    return res / dim[1]
end

function (A::AbstractLatticeFT)(x::Vararg{Number,NArgs}) where {NArgs}
    k = SVector(x)
    return A(k)
end

"""returns interpolated FT object

interpolatedFT(
    S_ab::AbstractMatrix{<:AbstractArray},
    BasisVectors::AbstractMatrix,
    UnitCellVectors::AbstractArray{<:AbstractArray},
    padding = AutomaticPadding(),
    plan = getLatticeFFTPlan(S_ab)
    kwargs,... 
)
kwargs are passed to getInterpolatedFFT: 
    Interpolation = BSpline(Cubic())
"""
function getLatticeFFT(
    S_ab::AbstractMatrix{<:AbstractArray},
    BasisVectors::AbstractMatrix,
    UnitCellVectors::AbstractArray{<:AbstractArray},
    padding=AutomaticPadding(),
    plan=getLatticeFFTPlan(S_ab, padding)
    ;kwargs...
)

    NCell, NCell2 = size(S_ab)
    @assert NCell == NCell2 "S_ab needs to be a square matrix"
    Sk_ab = [PhaseShiftedFFT(getInterpolatedFFT(S_ab[α, β], padding, plan; kwargs...), BasisVectors, UnitCellVectors[α] - UnitCellVectors[β]) for α in 1:NCell, β in 1:NCell]
    return LatticeFT(Sk_ab)
end

function getLatticeFFTPlan(S_ij::AbstractArray{<:Number}, padding=AutomaticPadding())
    return plan_fft(padSusc(S_ij, padding))
end

function getLatticeFFTPlan(S_ab::AbstractMatrix{<:AbstractArray}, padding=AutomaticPadding())
    return getLatticeFFTPlan(first(S_ab), padding)
end

