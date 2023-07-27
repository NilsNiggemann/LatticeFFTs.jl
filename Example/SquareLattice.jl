using SpinFRGLattices
import SpinFRGLattices.SquareLattice as SL
using FRGLatticeEvaluation
##
import FRGLatticeEvaluation as FLE
##
S = SL.getSquareLattice(5,[1,])
Lattice = LatticeInfo(S,SL)
CorrInfo = FLE.getCorrelationPairs(Lattice)
##
a = FLE.separateSublattices(CorrInfo.Ri_vec,CorrInfo.Rj_vec,-S.couplings[CorrInfo.pairs])
# aFT = FLE.getFFT.(a)
chi2k = FourierStruct(-S.couplings,Lattice)

FFTInt = FLE.interpolatedChi(a,SL.Basis,512)
##
""" given a matrix NxNxN return a 2D slice with the indices (i,i,j)"""
hhlslice(M) = M[[CartesianIndex(i,i,j) for i in axes(M,1), j in axes(M,3)]]
using CairoMakie,StaticArrays
let 
    fig = Figure()
    ax = Axis(fig[1,1],aspect = 1)
    k = LinRange(-2pi,2pi,100)
    chi = [real(sum(chiab(ki,kj) for chiab in FFTInt)) for ki in k, kj in k]
    chi2 = [chi2k(ki,kj) for ki in k, kj in k]
    hm = heatmap!(ax,k./pi,k./pi,chi)
    # hm = heatmap!(ax,k./pi,k./pi,chi2)
    Colorbar(fig[1,2],hm,ticks = -3:3)
    fig
end