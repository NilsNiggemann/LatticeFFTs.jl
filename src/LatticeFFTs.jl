module LatticeFFTs
    using FFTW, Interpolations

    """
    Given a function ChiR, return the Fourier transform via FFT.
    Assumptions:
        - ChiR is given in lattice coordinates, i.e. Chi[n1,n2,n3] corresponds to chi(0,n1*a1+n2*a2+n3*a3) 
        - The central index corresponds to chi(0,0). 
        - each dimension is of odd length
    """
    function getFFT(ChiR)
        @assert all(isodd.(dims(ChiR))) "works only for odd length of dimensions, but dims(ChiR) = $(dims(ChiR))"
        chik = real(fftshift(fft(ifftshift(chiR))))
        k = fftshift(fftfreq(N))*2π
        return (;k,chik)
    end

    function chi(alpha,beta,c)

    end
end # module LatticeFFTs
