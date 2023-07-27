module LatticeFFTs

    using FFTViews, Interpolations,StaticArrays
    using FFTW
    export interpolatedFT, padSusc, AutomaticPadding, LatticeFFT,getLatticeFFTPlan
    abstract type AbstractPadding end
    struct AutomaticPadding <: AbstractPadding end
    
    """
    Given a function χ(r) on a Bravais lattice, return a Fourier transform via FFT as a view.
    Assumptions:
        - χ(r) is given in lattice coordinates, i.e. Chi[n1,n2,n3] corresponds to chi(0,n1*a1+n2*a2+n3*a3) 
        - The central index corresponds to chi(0,0). 
    """
    function getFFT(ChiR,P = plan_fft(ChiR) )
        chik = FFTView(P* ifftshift(ChiR))
        return chik
    end

    """
    Pads the susceptibility to the final size 'newDims'. If any of the new dimensions is less than the original one, the old size is maintained
    """
    function padSusc(ChiR::AbstractArray{T},newDims::Tuple) where T <: Number
        dims = size(ChiR)
        newDims =  max.(dims,newDims)
        shift = CartesianIndex(( (d ≠ nd) && iseven(nd)  for (d,nd) in zip(dims,newDims))...)
    
        Origin = CartesianIndex((newDims .- dims).÷ 2 )
    
        # @info "" dims newDims shift Origin
        PaddedChiR = zeros(T,newDims)
        for I in CartesianIndices(ChiR)
            newI = I + shift + Origin
            PaddedChiR[newI] = ChiR[I]
        end
        return PaddedChiR
    end

    """
    Pads the susceptibility to the nearest power of 2 (minimum 64). This should ensure a fast computation and a smooth interpolation.
    """
    function padSusc(ChiR::AbstractArray{T},::AutomaticPadding) where T <: Number
        dims = size(ChiR)
        Nearest2Power = Tuple(ceil(Int,2^log2(d)) for d in dims)
        minSize = 64
        newDims =  Tuple(max(n,minSize) for n in Nearest2Power)
        return padSusc(ChiR,newDims)
    end
    
    function padSusc(ChiR::AbstractArray{T},val::Integer) where T <: Number
        val <= 0 && return ChiR
        padSusc(ChiR,Tuple(val for s in size(ChiR)))
    end
    
    function getInterpolatedFFT(Chi_ij::AbstractArray{<:Real},padding = AutomaticPadding(),args...)
        Chi_ij = padSusc(Chi_ij,padding)
        nk = Tuple(0:N for N in size(Chi_ij))
        FFT = getFFT(Chi_ij,args...)[nk...]
        k = Tuple(2π/N .* (0:N) for N in size(Chi_ij))
        chik = Interpolations.interpolate(k,FFT, Gridded(Linear()))
        chik = extrapolate(chik,Periodic(OnGrid()))
        return chik
    end
    
    abstract type AbstractPhaseShiftedFFT end
    
    struct PhaseShiftedFFT{InterpolationType,BasisMat<:AbstractMatrix,PhaseVecType<:AbstractVector} <: AbstractPhaseShiftedFFT
        S::InterpolationType
        T::BasisMat
        # UC::UCType
        PhaseVector::PhaseVecType
    end
    
    function (F::AbstractPhaseShiftedFFT)(k::AbstractVector)
        exp(1im*k'*F.PhaseVector)* F.S((F.T'*k)...)
    end

    function (F::AbstractPhaseShiftedFFT)(args...)
        k = SA[args...]
        return F(k)
    end
    
    import Base:size,getindex,setindex!,iterate,show,copy

    struct LatticeFFT{Mat<:AbstractMatrix{<:AbstractPhaseShiftedFFT}} 
        S::Mat
    end

    Base.getindex(S::LatticeFFT,i,j) = getindex(S.S,i,j)
    Base.setindex!(S::LatticeFFT,x,i,j) = setindex!(S.S,x,i,j)
    Base.iterate(S::LatticeFFT,i) = iterate(S.S,i)
    Base.iterate(S::LatticeFFT) = iterate(S.S)

    Base.size(S::LatticeFFT) = size(S.S)
    Base.copy(S::LatticeFFT) = LatticeFFT(copy(S.S))

    function (A::LatticeFFT)(k::AbstractVector)
        dim = size(A.S)
        return sum(a(k) for a in A)/dim[1]
    end
    function (A::LatticeFFT)(args...)
        k = SA[args...]
        return A(k)
    end
    """returns interpolated FT object

    interpolatedFT(
        S_ab::AbstractMatrix{<:AbstractArray},
        BasisVectors::AbstractMatrix,
        UnitCellVectors::AbstractArray{<:AbstractArray},
        padding = AutomaticPadding(),
        plan = getLatticeFFTPlan(S_ab)
    )
    """
    function interpolatedFT(
            S_ab::AbstractMatrix{<:AbstractArray},
            BasisVectors::AbstractMatrix,
            UnitCellVectors::AbstractArray{<:AbstractArray},
            padding = AutomaticPadding(),
            plan = getLatticeFFTPlan(S_ab,padding)
        )
        NCell,NCell2 = size(S_ab)
        @assert NCell == NCell2 "S_ab needs to be a square matrix"
        Sk_ab = [PhaseShiftedFFT(getInterpolatedFFT(S_ab[α,β],padding,plan),BasisVectors,UnitCellVectors[α] - UnitCellVectors[β]) for α in 1:NCell,β in 1:NCell]
        return LatticeFFT(Sk_ab)
    end

    function getLatticeFFTPlan(S_ij::AbstractArray{<:Number},padding = AutomaticPadding())
        return plan_fft(padSusc(S_ij,padding))
    end

    function getLatticeFFTPlan(S_ab::AbstractMatrix{<:AbstractArray},padding = AutomaticPadding())
        return getLatticeFFTPlan(first(S_ab),padding)
    end


end # module LatticeFFTs
