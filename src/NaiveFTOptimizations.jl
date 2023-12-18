struct RealPart{FType<:AbstractSubLatticeFT} <: AbstractSubLatticeFT
    FT::FType
end

struct ImagPart{FType<:AbstractSubLatticeFT} <: AbstractSubLatticeFT
    FT::FType
end

@inline (A::RealPart)(k::Vararg{Number,N}) where {N} = real(A.FT(k...))
@inline (A::ImagPart)(k::Vararg{Number,N}) where {N} = imag(A.FT(k...))

Base.imag(x::AbstractSubLatticeFT) = ImagPart(x)
Base.real(x::AbstractSubLatticeFT) = RealPart(x)

Base.imag(F::ImagPart) = F
Base.real(F::RealPart) = F

Base.real(F::ImagPart{T}) where T = zero(T)
Base.imag(F::RealPart{T}) where T = zero(T)

Base.real(x::AbstractLatticeFT) = LatticeFT(real.(x.S))
Base.imag(x::AbstractLatticeFT) = LatticeFT(imag.(x.S))


Base.:+(F::T, G::T) where {T<:RealPart} = RealPart(F.FT + G.FT)
Base.:*(G::RealPart,m::Number) = RealPart(G.FT*m)
Base.:*(m::Number,G::RealPart) = G*m
Base.:/(G::RealPart,m::Number) = RealPart(G.FT/m)

for (Reim,conversion,f) in zip((ImagPart,RealPart), (imag, real),(sin,cos))
    eval(:(
        @inline function (F::$Reim{<:naiveSubLatticeFT})(k::Vararg{AbstractFloat,2})
            (; Sij, Rij) = F.FT
            kx,ky = k
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
        @inline function (F::$Reim{<:naiveSubLatticeFT})(k::Vararg{AbstractFloat,3})
            kx,ky,kz = k
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
end