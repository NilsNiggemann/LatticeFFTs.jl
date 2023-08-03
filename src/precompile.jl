
@setup_workload begin
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

    
    @compile_workload begin
        (;a,b,Sij_ab) = testLattice
        ChiNaiveFast = LatticeFFTs.naiveLatticeFT(Sij_ab,a,b)
        Sq_ab = getLatticeFFT(Sij_ab,a,b,256)
    
    
        chis = [ChiNaiveFast(k,1.) for  k in LinRange(-4pi,4pi,20)]
        chis2 = [Sq_ab(k,1.) for  k in LinRange(-4pi,4pi,20)]
    end
end