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
function naiveFT(k,chiR,func)
    chik = 0.
    for n in eachindex(chiR)
        chik +=func(n)*cos(k*n)
    end
    return chik
end
function dressFT!(chiK)    
    for k in CartesianIndices(chiK)
        # chiK[k] *= exp(1im*pi*(sum(Tuple(k))))
        isodd(sum(Tuple(k))) && (chiK[k] *=-1)
    end
    return chiK
end
##
let 
    f1= 0.8*1pi
    N = 41
    Nreal = 9
    chi = OffsetArrays.centered(zeros(N))
    # chi = zeros(N)
    func2(i) = abs(i) > Nreal ? 0. : cos(f1*i)*exp(-(i)^2/40)
    func(i) = func2(i-1)
    chi = func.(eachindex(chi))

    
    k2 = LinRange(-2pi,2pi,1000)
    chiKCont = [naiveFT(k,chi,func2) for k in k2]
    chi = OffsetArrays.no_offset_view(chi)
    lines(chi) |> display
    # chiKfft = real(dressFT!(fft(chi)))
    # chiKfft = real(fft(fftshift(chi)))
    chiKfft = real(rfft(ifftshift(chi)))
    chiK = real(naiveDFT(chi))
    # chiK = naiveDFT(chi)
    k = rfftfreq(N)*2
    # push!(chiK,chiK[1])
    # k = 2piN*eachindex(chiK))
    # scatter(k,chiKfft)
    scatter(k,chiKfft)
    # scatter!(k,chiK)
    lines!(k2./pi,chiKCont,color = :black, linewidth = 5)
    # hlines!([8])
    # lines(k,OffsetArrays.no_offset_view(chiK))
    vlines!([f1/pi],color = :red)
    # xlims!(0,2*f2)
    chikextr = FLE.getInterpolatedFFT(chi)
    k = LinRange(-2pi,2pi,500)
    lines!(k./pi,real.(chikextr.(k)),color = :red)
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
""" given a matrix NxNxN return a 2D slice with the indices (i,i,j)"""
hhlslice(M) = M[[CartesianIndex(i,i,j) for i in axes(M,1), j in axes(M,3)]]
##
let 
    N = 101 # even and odd makes a difference of -1 phase!
    chiR = OffsetArrays.centered(zeros(N,N,N))
    # chiR = zeros(N,N)
    order = 0.9SA[1,1,1]*pi
    for ij in CartesianIndices(chiR)
        k = SVector(Tuple(ij))
        # cij = CubicAFMCorr(k)
        cij = Cubicspiral(k,order,N)
        chiR[ij] = cij
    end
    chiR = OffsetArrays.no_offset_view(chiR)

    # chiR = OffsetArrays.no_offset_view(chiR)
    # heatmap(collect(axes(chiR,1)),collect(axes(chiR,2)),chiR[:,:,1],axis = (;aspect=1)) |> display
    # chiK = fftshift(dressFT!(fft(chiR)))
    chiK = real(fftshift(fft(ifftshift(chiR))))
    # chiK = naiveDFT(chiR)
    # @info "chiK" maximum(imag(chiK))
    chiK = real(chiK)
    k = fftshift(fftfreq(N))*2
    fig = Figure()
    ax = Axis(fig[1,1],aspect = 1)
    # hm = heatmap!(ax,k,k,chiK[:,:,N÷2+1])
    hm = heatmap!(ax,k,k,hhlslice(chiK))
    # hm = heatmap!(ax,k,k,fftshift(chiK))
    scatter!(ax,Point2(order[1:2]...)/pi,markersize = 20,marker =  '×',color =:red)
    Colorbar(fig[1,2],hm)
    fig
end
##
let 
    k1 = LinRange(-pi,pi,50)
    y = sin.(k1)

    scatter(k1,y)
    yint = interpolate((k1,),y,Gridded(Linear()))
    yext = extrapolate(yint,Periodic())
    k2 = LinRange(-2pi,2pi,100)
    lines!(k2,yext.(k2))
    current_figure()
end