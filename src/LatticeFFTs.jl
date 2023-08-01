module LatticeFFTs


using FFTViews, Interpolations, StaticArrays, PaddedViews
using FFTW
using LoopVectorization

export getLatticeFFT, LatticeFT, getLatticeFFTPlan, getInterpolatedFFT, PhaseShiftedFFT, AbstractLatticeFT, naiveSubLatticeFT

abstract type AbstractSubLatticeFT end
abstract type AbstractLatticeFT{T<:AbstractSubLatticeFT} <: AbstractMatrix{T} end

include("FFT.jl")

include("NaiveFT.jl")
export naiveLatticeFT

end # module LatticeFFTs
