using SpinFRGLattices
import SpinFRGLattices.Pyrochlore as Py
using FRGLatticeEvaluation
using LatticeFFTs
##
import FRGLatticeEvaluation as FLE
##
# S = Py.getPyrochlore(4,[1,-0.5,0.3,-0.2,0.1])
S = Py.getPyrochlore(14,1 ./ 1:20)
Lattice = LatticeInfo(S,Py)
CorrInfo = FLE.getCorrelationPairs(Lattice)

a = FLE.separateSublattices(CorrInfo.Ri_vec,CorrInfo.Rj_vec,-S.couplings[CorrInfo.pairs])
# aFT = FLE.getFFT.(a)
chi2k = FourierStruct(-S.couplings,Lattice)

##
using CairoMakie,StaticArrays
function Pyrotest()
    fig = Figure()
    ax = Axis(fig[1,1],aspect = 1)
    k = LinRange(-8pi,8pi,400)
    @time begin
        FFTInt = FLE.interpolatedChi(a,Pyrochlore.Basis)
        chi = [real(sum(chiab(ki,ki,kj) for chiab in FFTInt))/Py.Basis.NCell for ki in k, kj in k]
    end
    @time chi2 = [chi2k(ki,ki,kj) for ki in k, kj in k]
    hm = heatmap!(ax,k,k,chi)
    # hm=heatmap!(ax,k,k,chi2)
    Colorbar(fig[1,2],hm)
    fig
end

Pyrotest()
Pyrotest()
