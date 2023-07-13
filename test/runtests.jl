using Test, LatticeFFTs,OffsetArrays
##
function naiveFT(k,chiR,func)
    chik = 0. +0im
    for n in eachindex(chiR)
        chik +=func(n)*exp(1im*k*n)
    end
    return chik
end

@testset "interpolated FFT" begin
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
            @test real(naiveFT(k,chi,func2)) ≈ real(chikextr(k)) atol = 1e-4
        end
    end
end