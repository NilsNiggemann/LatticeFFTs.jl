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
            chik += chiR[n+1]*exp(-1im*2pi*k*n/N)
            # chik += chiR[j]*cos(2pi*k'*n/N)
        end
        chiK[k+1] = real(chik)
        # chiK[i] = chiR[k]*cos(2pi*k'*k/N)
    end
    return chiK
end
function naiveFT(k,chiR)
    chik = 0.
    for n in 0:length(chiR)-1
        chik += chiR[n+1]*cos(k*n)
    end
    return chik
end
##
let 
    f1= 15.5
    f2 = 100
    N = 500
    # chi = OffsetArrays.centered(zeros(N))
    chi = zeros(N)
    for i in eachindex(chi)
        chi[i] = (-1)^(i)*exp(-abs(i)/5)
        # chi[i] = 1
    end
    # chi = OffsetArrays.no_offset_view(chi)

    # lines(collect(eachindex(chi)),chi) |> display
    
    chiKfft = real(fft(chi))
    chiK = real(naiveDFT(chi))
    k2 = LinRange(-pi,pi,1000)
    chiKCont = [naiveFT(k,chi) for k in k2]
    # chiK = naiveDFT(chi)
    k = fftfreq(N)*2
    # push!(chiK,chiK[1])
    # k = 2piN*eachindex(chiK))
    scatter(k,chiK)
    scatter!(k,chiKfft)
    lines!(k2./pi,chiKCont)
    # lines(k,OffsetArrays.no_offset_view(chiK))
    # vlines!([f1,f2],color = :red)
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
    N = 200
    chiR = OffsetArrays.centered(zeros(N,N))
    # chiR = zeros(N,N)
    order = SA[1.5,1.5]*2pi
    for ij in CartesianIndices(chiR)
        k = SVector(Tuple(ij))
        # cij = CubicAFMCorr(k)
        cij = Cubicspiral(k,order/N,N/2)
        chiR[ij] = cij
    end
    # chiR = OffsetArrays.no_offset_view(chiR)
    # heatmap(collect(axes(chiR,1)),collect(axes(chiR,2)),OffsetArrays.no_offset_view(chiR),axis = (;aspect=1)) |> display
    # chiK = ifft(chiR)*N
    chiK = naiveDFT(chiR)
    @info "chiK" maximum(imag(chiK))
    chiK = real(chiK)
    k = fftshift(fftfreq(N) )*N*2
    fig = Figure()
    ax = Axis(fig[1,1],aspect = 1)
    hm = heatmap!(ax,OffsetArrays.no_offset_view(chiK))
    # hm = heatmap!(ax,k,k,fftshift(chiK))
    scatter!(ax,Point2(order...)/pi)
    lims = 100pi
    limits!(ax,-lims,lims,-lims,lims)
    Colorbar(fig[1,2],hm)
    fig
end