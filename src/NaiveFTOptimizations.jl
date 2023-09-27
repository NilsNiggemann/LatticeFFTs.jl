struct RealPart{FType<:AbstractSubLatticeFT} <: AbstractSubLatticeFT
    FT::FType
end

struct ImagPart{FType<:AbstractSubLatticeFT} <: AbstractSubLatticeFT
    FT::FType
end

Base.imag(x::AbstractSubLatticeFT) = ImagPart(x)
Base.real(x::AbstractSubLatticeFT) = RealPart(x)

Base.imag(F::ImagPart) = F
Base.real(F::RealPart) = F

Base.real(F::ImagPart{T}) where T = zero(T)
Base.imag(F::RealPart{T}) where T = zero(T)

Base.real(x::AbstractLatticeFT) = LatticeFT(real.(x.S))


for (Reim,conversion,f) in zip((ImagPart,RealPart), (imag, real),(sin,cos))
    eval(:(
        @inline function (F::$Reim{<:naiveSubLatticeFT})(kx::AbstractFloat, ky::AbstractFloat)
            (; Sij, Rij) = F.FT
            Chi_k = zero(eltype(Sij))
            Rx, Ry = Rij

            @turbo for i in eachindex(Sij, Rx, Ry)
                rx = Rx[i]
                ry = Ry[i]

                res = $f(kx * rx + ky * ry)

                Chi_k += res * Sij[i]

            end

            return Chi_k
        end
    ))

    eval(:(
        @inline function (F::$Reim{<:naiveSubLatticeFT})(kx::AbstractFloat, ky::AbstractFloat, kz::AbstractFloat)
            (; Sij, Rij) = F.FT
            Chi_k = zero(eltype(Sij))
            Rx, Ry, Rz = Rij

            @turbo for i in eachindex(Sij, Rx, Ry)
                rx = Rx[i]
                ry = Ry[i]
                rz = Rz[i]

                res = $f(kx * rx + ky * ry + kz * rz)

                Chi_k += res * Sij[i]

            end

            return Chi_k
        end
        
    ))
    eval(:(
    @inline function (A::$Reim)(k::SVector)
        return A(k...)
    end
    ))
    eval(:(
    @inline function (A::$Reim)(k::Vararg{Number,N}) where {N}
        return $conversion(A.FT(k...))
    end
    ))
end
