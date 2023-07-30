module LatticeFFTs


using FFTViews, Interpolations, StaticArrays, PaddedViews
using FFTW
using LoopVectorization

export getLatticeFFT, LatticeFT, getLatticeFFTPlan, getInterpolatedFFT, AbstractLatticeFFT, PhaseShiftedFFT, AbstractLatticeFourierTransform, naiveSubLatticeFT

abstract type AbstractLatticeFourierTransform end
abstract type AbstractLatticeFFT <: AbstractLatticeFourierTransform end

include("FFT.jl")

include("NaiveFT.jl")
export naiveLatticeFT

end # module LatticeFFTs
