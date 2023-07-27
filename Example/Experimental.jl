using StaticArrays, CairoMakie
using OffsetArrays
import FRGLatticeEvaluation as FLE
using LatticeFFTs: fft, ifftshift, FFTViews
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
    f1= 1/3*1pi
    N = 32
    Nreal = 32
    chi = OffsetArrays.centered(zeros(N))
    # chi = zeros(N)
    func2(i) = abs(i) > Nreal ? 0. : cos(f1*i)*exp(-(i)^2/40)
    func(i) = func2(i)
    chi = func.(eachindex(chi))

    
    k = LinRange(105pi,106pi,500)
    chiKCont = [naiveFT(i,chi,func2) for i in k]
    chi = OffsetArrays.no_offset_view(chi)
    # lines(chi) |> display
    # chiKfft = rel(dressFT!(fft(chi)))
    # chiKfft = real(fft(fftshift(chi)))
    chiKfft = real(fft(LatticeFFTs.ifftshift(chi)))
    # chiKfftView = FFTView(fft(ifftshift(chi)))

    chiK = real(naiveDFT(chi))
    # chiK = naiveDFT(chi)
    # k = FFTViews.fftfreq(N)*2
    # push!(chiK,chiK[1])
    # k = 2piN*eachindex(chiK))
    # scatter(k,chiKfft)
    # scatter(k,chiKfft)
    # scatter!(k,chiK)
    lines(k./pi,chiKCont,color = :black, linewidth = 5)
    # hlines!([8])
    # lines(k,OffsetArrays.no_offset_view(chiK))
    # xlims!(0,2*f2)
    chikextr = LatticeFFTs.getInterpolatedFFT(chi,58)
    chi = reshape([chi,],1,1)
    # chikextr = interpolatedFT(chi,[1;;],[[0,]],64)
    lines!(k./pi,real.(chikextr.(k)),color = :red)
    # lines!(k./pi,real.(chikextr.(k*64/2pi)),color = :red)
    kBZ = -N:N
    # vlines!(f1/pi .* -1:2:6,color = :grey,linestyle = :dash)
    # lines!(2 .*kBZ ./N,real(chiKfftView[kBZ]),color = :red)
    
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
function applyFunc(k,chik)
    chi = zeros(length(k),length(k))
    @time for (i,kx) in enumerate(k), (j,ky) in enumerate(k)
        chi[i,j] = chik(kx,ky,0.)
    end
end
##
function testCubic()
    N = 60 +1 # even and odd makes a difference of -1 phase!
    chiR = OffsetArrays.centered(zeros(N,N,3N))
    # chiR = zeros(N,N)
    order = 0.4SA[1,1,0]*pi
     
    @time for ij in CartesianIndices(chiR)
        k = SVector(Tuple(ij))
        # cij = CubicAFMCorr(k)
        cij = Cubicspiral(k,order,2N)
        chiR[ij] = cij
    end
    chiR = OffsetArrays.no_offset_view(chiR)

    # chiR = OffsetArrays.no_offset_view(chiR)
    # heatmap(collect(axes(chiR,1)),collect(axes(chiR,2)),chiR[:,:,1],axis = (;aspect=1)) |> display
    # chiK = fftshift(dressFT!(fft(chiR)))
    # chiK = real(fftshift(fft(ifftshift(chiR))))
    # chiK = naiveDFT(chiR)
    # @info "chiK" maximum(imag(chiK))
    # chiK = real(chiK)

    # k = fftshift(fftfreq(N))*2
    @time chik = FLE.getInterpolatedFFT(chiR,128)
    # return chik
    k = LinRange(0,8pi,500)
    # @time chi = [chik(kx,ky,0.) for kx in k, ky in k]
    @time chi = applyFunc(k,chik)
    return
    fig = Figure()
    ax = Axis(fig[1,1],aspect = 1)
    # hm = heatmap!(ax,k,k,chiK[:,:,N÷2+1])
    # hm = heatmap!(ax,k,k,hhlslice(chiK))
    hm = heatmap!(ax,k,k,chi)
    # hm = heatmap!(ax,k,k,fftshift(chiK))
    ps = [Point(-pi,-pi),Point(pi,-pi),Point(pi,pi),Point(-pi,pi),Point(-pi,-pi)]
    lines!([Point2(2pi,2pi) + p for p in ps])
    scatter!(ax,[Point2(order[1:2]...), Point2(6pi .+order[1:2]...)],markersize = 20,marker =  '×',color =:red)
    Colorbar(fig[1,2],hm)
    fig
end
##
testCubic()
@profview testCubic()
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