using LatticeFFTs
##
using Test,OffsetArrays, StaticArrays
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
    @testset "$(("even","odd")[shift+1])" for shift in (0,1)
        N = 20+1
        chi = OffsetArrays.centered(zeros(N))

        func2(i) = cos(f1*i)*exp(-(i)^2/40)
        func(i) = func2(i - iseven(N)) #shift center for even N

        chi = func.(eachindex(chi))
        
        chiArray = OffsetArrays.no_offset_view(chi)

        chikextr = LatticeFFTs.getInterpolatedFFT(chiArray,512)

        for k in (f1,pi,0,0.13)
            @test real(naiveFT(SA[k],chi,func2)) ≈ real(chikextr(k)) atol = 1e-6
        end
    end
end

##
function Cubicspiral(n,k,xi=10000)
    return (cos(k'*n)) *exp(-n'*n/xi)
end

@testset "Cubic" begin
    @testset "$(("even","odd")[shift+1])" for shift in (0,1)
        N = 40
        chiR = OffsetArrays.centered(zeros(N,N,3N))
        # chiR = zeros(N,N)
        order = 0.835SA[1,1,0]*pi
        Config(n) = Cubicspiral(SA[Tuple(n)...],order,30)
        Config(n1,n2,n3) = Cubicspiral(SA[n1,n2,n3],order,30)
        
        for ij in CartesianIndices(chiR)
            k = SVector(Tuple(ij))
            # cij = CubicAFMCorr(k)
            cij = Config(ij)
            chiR[ij] = cij
        end
        chiRArray = OffsetArrays.no_offset_view(chiR)
        
        @time chik = getInterpolatedFFT(chiRArray,128)

        chikNaive(k) = real(naiveFT(k,chiR,Config))
        chikFFT(k) = real(chik(k...))

        @test chikNaive(order) ≈ chikFFT(order) atol = 1e-2
        @testset "improve accuracy at incommensurate point" begin
            @test chikNaive(order) ≈ chikFFT(order) atol = 1e-3 broken = true
        end
        @test chikNaive(SA[pi,pi,pi]) ≈ chikFFT(SA[pi,pi,pi]) atol = 1e-13
        @test chikNaive(SA[0,0,0]) ≈ chikFFT(SA[0,0,0]) atol = 1e-13
        @test chikNaive(SA[0.13,0.5,√2]) ≈ chikFFT(SA[0.13,0.5,√2]) atol = 1e-13
    end
end
##


function ComplexFourier(Rij_vec,Sij_vec,k)
    Chi_k = 0
    for (rij,sij) in zip(Rij_vec,Sij_vec)
        Chi_k += exp(1im*(k' *rij)) * sij
    end
    return Chi_k
end



function getHoneycombTest()
    J1 = -1.0
    J2 = -0.4

    L = 2

    a1 = SA[1,0]

    a2 = SA[-1/2,√(3/4)]

    b = [SA[0,0],SA[0,1/√3]]

    cart(Ri) = Ri[1]*a1 + Ri[2]*a2 + b[Ri[3]]

    function dist(Ri,Rj) 
        rij = cart(Ri)-cart(Rj)
        return sqrt(rij' * rij)
    end

    Lattice = [(;n1,n2,b) for n1 in -L:L for n2 in -L:L for b in 1:2]
    UC =  [(;n1=0,n2=0,b=1),(;n1=0,n2=0,b=2)]
    

    distances = [round(dist(Ri,Rj),digits = 10) for Ri in UC for Rj in Lattice] |> sort! |> unique!

    function S(Ri,Rj)

        d = dist(Ri,Rj)
        # return 1/d^2
        if d ≈ distances[2] atol = 1e-4
            return J1
        elseif d ≈ distances[3] atol = 1e-4
            return J2
        else
            return 0.
        end
    end

    Sij(a,b) = [S((0,0,a),(;n1,n2,b)) for n1 in -L:L, n2 in -L:L]

    Sij_ab = [Sij(a,b) for a in 1:2, b in 1:2]
    
    Rij_vec_(a,b) = [(cart(Ri)-cart(Rj)) for Ri in UC for Rj in Lattice if Ri.b == a && Rj.b == b]
    Sij_vec_(a,b) = [S(Ri,Rj) for Ri in UC for Rj in Lattice if Ri.b == a && Rj.b == b]
    
    Rij_vec = [Rij_vec_(a,b) for a in 1:2, b in 1:2]
    Sij_vec = [Sij_vec_(a,b) for a in 1:2, b in 1:2]

    return (;a = [a1 a2] ,b,Sij_ab,Rij_vec,Sij_vec)
end 

testLattice = getHoneycombTest()
##
function testLatticeFFT(Sq_ab,chiNaive)
         
    Points = Dict([
        "Γ" => SVector(0,0),
        "K" => SVector(4π/3,0),
        "M" => SVector(π, -π/√3),
        "(π,π)" => SVector(π,π),
        "(√(22),100/3" => SVector(√(22),100/3)
    ]
    )

    @testset "diag real" begin
        @testset "k = $key" for (key,k) in Points
            Sq_11 = Sq_ab[1,1](k)
            Sq_22 = Sq_ab[2,2](k)
            @test imag(Sq_11) ≈ 0 atol = 1e-14
            @test imag(Sq_22) ≈ 0 atol = 1e-14
        end
    end
    @testset "sublattices equivalent" begin
        @testset "k = $key" for (key,k) in Points
            @test Sq_ab[1,1](k) ≈ Sq_ab[2,2](k) atol = 1e-14
            @test Sq_ab[1,2](k) ≈ Sq_ab[2,1](k)' atol = 1e-14
        end
    end
    
    @testset "Sublattice average" begin
        @testset "k = $key" for (key,k) in Points
            Sq = sum(sq(k) for sq in Sq_ab)/size(Sq_ab)[1]
            @test Sq_ab(k) ≈ Sq atol = 1e-14
        end
    end
    @testset "Check against naive implementation" begin
        @testset "k = $key" for (key,k) in Points
            @test Sq_ab[1,1](k) ≈ chiNaive[1,1](k) atol = 1e-8
            @test Sq_ab[1,2](k) ≈ chiNaive[1,2](k) atol = 1e-8
            @test Sq_ab[2,2](k) ≈ chiNaive[2,2](k) atol = 1e-8
        end
    end

end

@testset "Non-Bravais: naive" verbose = true begin
    (;a,b) = testLattice

    ChiNaiveFast = LatticeFFTs.naiveLatticeFT(testLattice.Sij_ab,a,b)

    chiNaive = [k-> ComplexFourier(testLattice.Rij_vec[a,b],testLattice.Sij_vec[a,b],k) for a in 1:2, b in 1:2]

    testLatticeFFT(ChiNaiveFast,chiNaive)
end

@testset "math ops" begin
    (;a,b) = testLattice

    ChiNaiveFast = LatticeFFTs.naiveLatticeFT(testLattice.Sij_ab,a,b)
    
    @testset "real and imag" begin
        ChiRe = real(ChiNaiveFast)
        ChiIm = imag(ChiNaiveFast)
        @test ChiNaiveFast(1.,2) ≈ ChiRe(1.,2) 
        @test real(ChiNaiveFast[1,2](1.,2)) ≈ ChiRe[1,2](1.,2)
        @test imag(ChiNaiveFast[1,2](1.,2)) ≈ ChiIm[1,2](1.,2)
    end
    @testset "addition and multiplication" begin
        chidiag = real(sum(ChiNaiveFast[i,i] for i in axes(ChiNaiveFast,1)))
        @test length(chidiag.FT.Sij) == length(b)* length(ChiNaiveFast[1,1].Sij)

        chioffdiag = real(sum(ChiNaiveFast[i,j] for i in axes(ChiNaiveFast,1), j in axes(ChiNaiveFast,2) if j<i))

        chiTot = (chidiag + 2*chioffdiag) /length(b)
        @test chiTot(1.,2) ≈ ChiNaiveFast(1.,2) atol = 1e-14

    end
end
##
@testset "Non-Bravais FFT" verbose = true begin
    (;a,b) = testLattice
    Sq_ab = getLatticeFFT(testLattice.Sij_ab,a,b,256)
    ChiNaiveFast = LatticeFFTs.naiveLatticeFT(testLattice.Sij_ab,a,b)

    chiNaive = [k-> ComplexFourier(testLattice.Rij_vec[a,b],testLattice.Sij_vec[a,b],k) for a in 1:2, b in 1:2]


    testLatticeFFT(Sq_ab,chiNaive)

end
