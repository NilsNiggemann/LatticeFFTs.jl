using StaticArrays, CairoMakie
using OffsetArrays
using LatticeFFTs.FFTW
##
function naiveDFT(chiR)
    N = size(chiR,1)
    chiK = similar(chiR)
    for k in 0:N-1
        chik = 0. +0im
        for n in 0:N-1
            chik += chiR[n+1]*exp(-1im*2pi*k*n/N)*exp(1im*pi*(k+1))
            # chik += chiR[j]*cos(2pi*k'*n/N)
        end
        chiK[k+1] = real(chik)
        # chiK[i] = chiR[k]*cos(2pi*k'*k/N)
    end
    return chiK
end
function naiveFT(k,chiR)
    chik = 0.
    N = length(chiR)-1
    for n in eachindex(chiR)
        chik +=(-1)^n*exp(-n^2/20)*cos(k*n)
        # @info "test" n chiR[abs(n)+1] (-1)^n*exp(-n^2/20)
        # chik += chiR[n]*cos(k*n)
    end
    return chik
end
function dressFT!(chiK)    
    for k in CartesianIndices(chiK)
        # chiK[k] *= exp(1im*pi*sum(Tuple(k)))
        isodd(sum(Tuple(k))) && (chiK[k] *=-1)
    end
    return chiK
end
##
let 
    f1= 0.5*1pi
    N = 401
    chi = OffsetArrays.centered(zeros(N))
    # chi = zeros(N)
    for i in eachindex(chi)
        chi[i] = (-1)^(i)*exp(-(i)^2/20)
        # chi[i] = cos(f1*i)*exp(-(i)^2/40)
        # chi[i] = 1
    end
    k2 = LinRange(-pi,pi,100)
    chiKCont = [naiveFT(k,chi) for k in k2]
    chi = OffsetArrays.no_offset_view(chi)[1:end-1]
    N-=1
    lines(collect(eachindex(chi)),chi) |> display
    chiKfft = real(dressFT!(fft(chi)))
    chiK = real(naiveDFT(chi))
    # chiK = naiveDFT(chi)
    k = fftfreq(N)*2
    # push!(chiK,chiK[1])
    # k = 2piN*eachindex(chiK))
    scatter(k,chiK)
    # scatter!(k,chiKfft)
    lines!(k2./pi,chiKCont)
    hlines!([8])
    # lines(k,OffsetArrays.no_offset_view(chiK))
    vlines!([f1/pi],color = :red)
    # xlims!(0,2*f2)
    current_figure()
end
##

function CubicAFMCorr(n)
    return (-1)^(sum(n))*exp(-n'*n/100)
end
function Cubicspiral(n,k,xi=10000)
    return (cos(k'*n)) *exp(-n'*n/xi)
end

##

let 
    N = 101
    chiR = OffsetArrays.centered(zeros(N,N))
    # chiR = zeros(N,N)
    order = 0.2SA[1,1]*pi
    for ij in CartesianIndices(chiR)
        k = SVector(Tuple(ij))
        # cij = CubicAFMCorr(k)
        cij = Cubicspiral(k,order,N)
        chiR[ij] = cij
    end
    chiR = OffsetArrays.no_offset_view(chiR)

    # chiR = OffsetArrays.no_offset_view(chiR)
    heatmap(collect(axes(chiR,1)),collect(axes(chiR,2)),chiR[:,:,1],axis = (;aspect=1)) |> display
    chiK = fftshift(dressFT!(fft(chiR)))
    # chiK = naiveDFT(chiR)
    # @info "chiK" maximum(imag(chiK))
    chiK = real(chiK)
    k = fftshift(fftfreq(N))*2
    fig = Figure()
    ax = Axis(fig[1,1],aspect = 1)
    hm = heatmap!(ax,k,k,chiK[:,:,1])
    # hm = heatmap!(ax,k,k,fftshift(chiK))
    scatter!(ax,Point2(order[1:2]...)/pi,markersize = 20,marker =  '×',color =:red)
    Colorbar(fig[1,2],hm)
    fig
end