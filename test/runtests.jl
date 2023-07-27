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
# @testset "Non-Bravais" begin


function ComplexFourier(Rij_vec,Sij_vec,k)
    Chi_k = 0
    for (rij,sij) in zip(Rij_vec,Sij_vec)
        Chi_k += exp(1im*(k' *rij)) * sij
    end
    return Chi_k
end

# @testset "Non-Bravais" verbose = true begin
let
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
    
    Sq_ab = interpolatedFT(Sij_ab,[a1 a2],b,256)
    k = LinRange(-8pi,8pi,100)
    
    α = 1
    β = 2

    Rij_vec(a,b) = [(cart(Ri)-cart(Rj)) for Ri in UC for Rj in Lattice if Ri.b == a && Rj.b == b]
    Sij_vec(a,b) = [S(Ri,Rj) for Ri in UC for Rj in Lattice if Ri.b == a && Rj.b == b]
    chiNaive(k::SVector,a,b) = ComplexFourier(Rij_vec(a,b),Sij_vec(a,b),k)
    chiNaive(k,a,b) = chiNaive(SA[k...],a,b)
     
    Points = Dict([
        "Γ" => (0,0),
        "K" => (4π/3,0),
        "M" => (π, -π/√3),
        "(π,π)" => (π,π),
        "(√(22),100/3" => (√(22),100/3)
    ]
    )

    @testset "diag real" begin
        @testset "k = $key" for (key,k) in Points
            Sq_11 = Sq_ab[1,1](k...)
            Sq_22 = Sq_ab[2,2](k...)
            @test imag(Sq_11) ≈ 0 atol = 1e-14
            @test imag(Sq_22) ≈ 0 atol = 1e-14
        end
    end
    @testset "sublattices equivalent" begin
        @testset "k = $key" for (key,k) in Points
            @test Sq_ab[1,1](k...) ≈ Sq_ab[2,2](k...) atol = 1e-14
            @test Sq_ab[1,2](k...) ≈ Sq_ab[2,1](k...)' atol = 1e-14
        end
    end
    @testset "Check against naive implementation" begin
        @testset "k = $key" for (key,k) in Points
            @test Sq_ab[1,1](k...) ≈ chiNaive(k,1,1) atol = 1e-8
            @test Sq_ab[1,2](k...) ≈ chiNaive(k,1,2) atol = 1e-8
            @test Sq_ab[2,2](k...) ≈ chiNaive(k,2,2) atol = 1e-8
        end
    end
end