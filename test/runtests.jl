using Test,OffsetArrays, StaticArrays
using LatticeFFTs
##
function naiveFT(k::AbstractVector,chiR,func)
    chik = 0. +0im
    for n in CartesianIndices(chiR)
        nVec = SA[Tuple(n)...]
        chik +=func(nVec...)*exp(1im*k'*nVec)
    end
    return chik
end

@testset "1D chain" begin
    f1= 0.8*1pi
    for shift in (0,1)
        N = 3002+1
        chi = OffsetArrays.centered(zeros(N))

        func2(i) = cos(f1*i)*exp(-(i)^2/40)
        func(i) = func2(i - iseven(N)) #shift center for even N

        chi = func.(eachindex(chi))
        
        chiArray = OffsetArrays.no_offset_view(chi)

        chikextr = LatticeFFTs.getInterpolatedFFT(chiArray,512)

        for k in (f1,pi,0,0.13)
            @test real(naiveFT(SA[k],chi,func2)) ≈ real(chikextr(k)) atol = 1e-4
        end
    end
end

function Cubicspiral(n,k,xi=10000)
    return (cos(k'*n)) *exp(-n'*n/xi)
end
##
@testset "Cubic" begin
    N = 40 +1
    chiR = OffsetArrays.centered(zeros(N,N,3N))
    # chiR = zeros(N,N)
    order = 0.5SA[1,1,0]*pi
    Config(n) = Cubicspiral(SA[Tuple(n)...],order,30)
    Config(n1,n2,n3) = Cubicspiral(SA[n1,n2,n3],order,30)
     
    for ij in CartesianIndices(chiR)
        k = SVector(Tuple(ij))
        # cij = CubicAFMCorr(k)
        cij = Config(ij)
        chiR[ij] = cij
    end
    chiRArray = OffsetArrays.no_offset_view(chiR)
    
    @time chik = LatticeFFTs.getInterpolatedFFT(chiRArray,128)

    chikNaive(k) = real(naiveFT(k,chiR,Config))
    chikFFT(k) = real(chik(k...))

    @test chikNaive(order) ≈ chikFFT(order) atol = 1e-4
    @test chikNaive(SA[pi,pi,pi]) ≈ chikFFT(SA[pi,pi,pi]) atol = 1e-4
    @test chikNaive(SA[0,0,0]) ≈ chikFFT(SA[0,0,0]) atol = 1e-4
    @test chikNaive(SA[0.13,0.5,√2]) ≈ chikFFT(SA[0.13,0.5,√2]) atol = 1e-4
end