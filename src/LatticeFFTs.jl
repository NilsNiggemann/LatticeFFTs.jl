module LatticeFFTs


using FFTViews, Interpolations, StaticArrays, PaddedViews
using FFTW
using LoopVectorization

export getLatticeFFT, LatticeFT, getLatticeFFTPlan, getInterpolatedFFT, PhaseShiftedFFT, AbstractLatticeFT, naiveSubLatticeFT

abstract type AbstractSubLatticeFT end
abstract type AbstractLatticeFT{T<:AbstractSubLatticeFT} <: AbstractMatrix{T} end


Base.size(S::AbstractLatticeFT) = size(S.S)
Base.getindex(S::AbstractLatticeFT, I::Vararg{Int, N}) where N	= getindex(S.S, I...)
Base.setindex!(S::AbstractLatticeFT, x, I::Vararg{Int, N}) where N = setindex!(S.S, x, I...)



include("FFT.jl")

include("NaiveFT.jl")
export naiveLatticeFT

using PrecompileTools
include("precompile.jl")

end # module LatticeFFTs
