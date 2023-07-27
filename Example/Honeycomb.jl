using SpinFRGLattices
import SpinFRGLattices.Honeycomb as Ho
using FRGLatticeEvaluation
##
import FRGLatticeEvaluation as FLE
##
# S = Py.getPyrochlore(4,[1,-0.5,0.3,-0.2,0.1])
S = Ho.getHoneycomb(14,[1,0.4])
Lattice = LatticeInfo(S,Ho)
CorrInfo = FLE.getCorrelationPairs(Lattice)
# S.couplings[1] = -3
a = FLE.separateSublattices(CorrInfo.Ri_vec,CorrInfo.Rj_vec,-S.couplings[CorrInfo.pairs])
# a[2,1] .= a[2,1]'

# aFT = FLE.getFFT.(a)
chi2k = FourierStruct(-S.couplings,Lattice)

function TestHoneycombBasis()
    a1 = SA[1,0]
    a2 = SA[-1/2,√(3/4)]
    
    b0 = SA[0,0]
    b1 = 0*SA[1,0]
    # b1 = 0*Ho.Basis.b[2]
    return Basis_Struct_2D(a1=a1,a2=a2,b=[b0,b1],NNdist = norm(b1))
end
HoBasis2 = Ho.Basis


function FTNaive(k,Rivec,Rjvec,Sij)
    chik = 0. +0im
    for i in eachindex(Rivec,Rjvec,Sij)
        ri = Rivec[i]
        rj = Rjvec[i]
        chik += exp(-1im*k'*(ri-rj))*Sij[i]
        # chik += cos(k'*(ri-rj))*Sij[i]
    end
    return chik
end



function reducedCorr(CorrInfo,a,b) 
    inds = [i for (i,(ri,rj)) in enumerate(zip(CorrInfo.Ri_vec,CorrInfo.Rj_vec)) if ri.b == a && rj.b == b && S.couplings[CorrInfo.pairs[i]] != 0]
    Ri = getCartesian.(CorrInfo.Ri_vec[inds],Ref(HoBasis2))
    Rj = getCartesian.(CorrInfo.Rj_vec[inds],Ref(HoBasis2))
    Rijvec = [ri .- rj for (ri,rj) in zip(Ri,Rj)]
    Sij = -S.couplings[CorrInfo.pairs[inds]]
    return (;Ri,Rj,Sij)
    # FourierStruct(-S.couplings[CorrInfo.pairs[inds]],Rijvec,HoBasis2.NCell)
end

##
function chiab(a,b) 
    inds = [i for (i,(ri,rj)) in enumerate(zip(CorrInfo.Ri_vec,CorrInfo.Rj_vec)) if ri.b == a && rj.b == b]
    Ri = getCartesian.(CorrInfo.Ri_vec[inds],Ref(HoBasis2))
    Rj = getCartesian.(CorrInfo.Rj_vec[inds],Ref(HoBasis2))
    Rijvec = [ri .- rj for (ri,rj) in zip(Ri,Rj)]
    Sij = -S.couplings[CorrInfo.pairs[inds]]
    return (args...)->FTNaive(SA[args...],Ri,Rj,Sij)
    # FourierStruct(-S.couplings[CorrInfo.pairs[inds]],Rijvec,HoBasis2.NCell)
end



##
FFTInt = FLE.interpolatedChi(a,Ho.Basis,65)

using CairoMakie,StaticArrays
let 
    # fig = Figure()
    fig = Figure(resolution = 400 .*(1.2,2))
    ax1 = Axis(fig[1,1],aspect = 1)
    ax2 = Axis(fig[2,1],aspect = 1)
    k = LinRange(-8pi,8pi,300)
    
    α = 1
    β = 2

    chi = [real(FFTInt[α,β](ki,kj)) for ki in k, kj in k]
    chi2 = [real(chiab(α,β)(ki,kj)) for ki in k, kj in k]
    # chi2 = [real( chiab(α,β)(ki,kj)) for ki in k, kj in k]
    hm = heatmap!(ax1,k,k,chi)
    hm2 = heatmap!(ax2,k,k,chi2)
    Colorbar(fig[1,2],hm)
    Colorbar(fig[2,2],hm2)
    fig
end
##

let 
    fig = Figure()
    ax = Axis(fig[1,1],aspect = 1)
    k = LinRange(-8pi,8pi,100)
    chi = [real(FFTInt(ki,kj)) for ki in k, kj in k]
    chi2 = [chi2k(ki,kj) for ki in k, kj in k]
    hm = heatmap!(ax,k,k,chi)
    hm = heatmap!(ax,k,k,chi2)
    Colorbar(fig[1,2],hm)
    fig
end