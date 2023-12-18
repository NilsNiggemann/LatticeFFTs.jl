using SpinFRGLattices
import SpinFRGLattices.Pyrochlore as Py
using FRGLatticeEvaluation
using LatticeFFTs
##
import FRGLatticeEvaluation as FLE
##
# S = Py.getPyrochlore(4,[1,-0.5,0.3,-0.2,0.1])
S = Py.getPyrochlore(5,[1,])
Lattice = LatticeInfo(S,Py)
CorrInfo = FLE.getCorrelationPairs(Lattice)

a = FLE.separateSublattices(CorrInfo.Ri_vec,CorrInfo.Rj_vec,-S.couplings[CorrInfo.pairs])
##
# aFT = FLE.getFFT.(a)
chi2k = FourierStruct(-S.couplings,Lattice)

@inline function ComplexFourier(F::FourierStruct3D,kx::T,ky::T,kz::T) where T <: Real
    Chi_k = zero(T)
    (;Rx_vec,Ry_vec,Rz_vec,Sij_vec,NCell) = F
    for i in eachindex(Rx_vec,Ry_vec,Rz_vec,Sij_vec)
        Rx = Rx_vec[i]
        Ry = Ry_vec[i]
        Rz = Rz_vec[i]
        Chi_k += exp(1im*(kx*Rx+ky*Ry+kz*Rz)) * Sij_vec[i]
    end
    return Chi_k/NCell
end


##
using CairoMakie,StaticArrays
function Pyrotest()
    fig = Figure(resolution = (500,800))
    ax = Axis(fig[1,1],aspect = 1)
    ax2 = Axis(fig[2,1],aspect = 1)
    k = LinRange(-8pi,8pi,90)
    @time begin
        FFTInt = FLE.interpolatedChi(a,Pyrochlore.Basis,128)
        # chi = [real(sum(chiab(ki,ki,kj) for chiab in FFTInt))/Py.Basis.NCell for ki in k, kj in k]
        chi = [real(FFTInt(ki,ki,kj)) for ki in k, kj in k]
    end
    @time chi2 = [chi2k(ki,ki,kj) for ki in k, kj in k]
    # @time chi2 = [ComplexFourier(chi2k,ki,ki,kj) for ki in k, kj in k]
    hm = heatmap!(ax,k,k,chi)
    hm2 = heatmap!(ax2,k,k,chi2)
    Colorbar(fig[1,2],hm)
    Colorbar(fig[2,2],hm2)
    fig
end

Pyrotest()
# Pyrotest()
##
@profview FLE.interpolatedChi(a,Pyrochlore.Basis,128)