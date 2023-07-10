using SpinFRGLattices
import SpinFRGLattices.Pyrochlore as Py
using FRGLatticeEvaluation
##
import FRGLatticeEvaluation as FLE
##
S = Py.getPyrochlore(6)
Lattice = LatticeInfo(S,Py)
CorrInfo = FLE.getCorrelationPairs(Lattice)
##
a = FLE.separateSublattices(CorrInfo.Ri_vec,CorrInfo.Rj_vec,-S.couplings[CorrInfo.pairs])
# aFT = FLE.getFFT.(a)
FFTInt = FLE.interpolatedChi(a,Pyrochlore.Basis)
##
""" given a matrix NxNxN return a 2D slice with the indices (i,i,j)"""
hhlslice(M) = M[[CartesianIndex(i,i,j) for i in axes(M,1), j in axes(M,3)]]
using CairoMakie,StaticArrays
let 
    fig = Figure()
    ax = Axis(fig[1,1],aspect = 1)
    k = LinRange(-1.3,1,100)
    chi = [real(sum(chiab(SA[ki,ki,kj]) for chiab in FFTInt)) for ki in k, kj in k]
    heatmap!(ax,k,k,chi)
    fig
end