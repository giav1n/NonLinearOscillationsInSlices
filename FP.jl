#Fokker-Planck integration for 1D integrate and fire neurons
# Gianni V. Vinci 05/2024

module FP
using LinearAlgebra,HDF5,SparseArrays,PoissonRandom,DelimitedFiles

#Single Population fixed parameters
struct SinglePopulation
    #Integration time step
    dt::Float64
    #Membrane decay time
    τ::Float64
    #Absolute refractory period
    τ0::Float64
    #Mean delay time
    τD::Float64
    #Adaptation time scale
    τC::Float64
    #Threshold and reset potential
    θ::Float64
    H::Float64
    dV::Float64
    V::Vector{Float64}
    NV::Int
    ib::Int
    nref::Int
    nδ::Int
    g::Float64
    N::Int

end

#Initilize distribution and grid of integration:
function GridInitialization(Vmin,H,θ,NV)

    VC=Array(range(Vmin,stop=θ,length=NV))
    dVCenters=(VC[end]-VC[end-1])/2;

    VC=VC .- dVCenters;
    dVInterfaces=diff(VC);
    VI=zeros(NV+1)

    VI[2:end-1]=VC[1:NV-1] + (0.5).*dVInterfaces
    VI[end]=VC[end] + (0.5).* dVInterfaces[end];
    VC[1]=VC[1] - (0.5).* dVInterfaces[1];
    dV=VC[3]-VC[2]
    ib=argmin(abs.(VC.-H)) #reinjection index
    return VI,VC,ib,dV
end



function DefineSinglePopulation(p)
    Vmin=p["Vm"]
    NV=p["NV"]
    V,Vc,ib,dV=GridInitialization(Vmin,p["H"],p["θ"],NV)

    nref=Int(round(p["τ0"]/p["dt"]))
    nδ=Int(round(p["δ"]/p["dt"]))


    cm=SinglePopulation( p["dt"],p["τ"],p["τ0"],p["τD"],
                         p["τC"],p["θ"],p["H"],dV,V,NV,
                         ib,nref,nδ,p["g"],p["N"])
    return cm
end

function InitializeState(cm)
    S=zeros(cm.NV +3 +max(cm.nref,cm.nδ))
    #Define State
    S[cm.ib]=1/cm.dV  # default initialization is p=δ(v-H)
    S[cm.NV+1]=0.0    #  ν
    S[cm.NV+2]=0.0    # νd
    S[cm.NV+3]=0.0    # c


    return S
end

function exp_vdV_D(v,dV,D)
    return exp(-v*dV/D)
end


# Deterministic component of LIF neural model
function GetF!(f,V,μ,τ,El,Nv)
    @inbounds for i=1:Nv
        f[i]=(El-V[i])/τ +μ
    end
end


# Diagonals of matrix rapresentation of the Fokker-Planck evolution operator:
function Diagonals!(mat,N,v,D,dV,dt)
    dt_dV = dt/dV
    @inbounds for i=2:N-1
        if (v[i] != 0.0)
            exp_vdV_D1 = exp_vdV_D(v[i],dV,D)
            mat[2,i] = -dt_dV*v[i]*exp_vdV_D1/(1.0 -exp_vdV_D1) # diagonal
            mat[3,i-1] = dt_dV*v[i]/(1.0 -exp_vdV_D1) # lower diagonal
        else
            mat[1,i] = -dt_dV*D/dV # diagonal
            mat[2,i-1] = dt_dV*D/dV # lower diagonal
        end

        if (v[i+1]!=0.0)
            exp_vdV_D2 = exp_vdV_D(v[i+1],dV,D)
            mat[2,i] -= dt_dV*v[i+1]/(1.0-exp_vdV_D2) # diagonal
            mat[1,i+1] = dt_dV*v[i+1]*exp_vdV_D2/(1.0 -exp_vdV_D2) # upper diagonal
        else
            mat[2,i] -= dt_dV*D/dV # diagonal
            mat[1,i+1] = dt_dV*D/dV # upper diagonal
        end
    end

    # Boundary conditions
    if (v[2] != 0.0)
        tmp1 = v[2]/(1.0 -exp_vdV_D(v[2],dV,D))
    else
        tmp1 = D/dV
    end

    if (v[end] != 0.0)
        tmp2 = v[end]/(1.0 -exp_vdV_D(v[end],dV,D))
    else
        tmp2 = D/dV
    end
    if (v[end-1] != 0.0)
        tmp3 = v[end-1]/(1.0 -exp_vdV_D(v[end-1],dV,D))
    else
        tmp3 = D/dV
    end

    mat[2,1] = -dt_dV*tmp1                      # first diagonal
    mat[1,2] = dt_dV*tmp1*exp_vdV_D(v[2],dV,D)  # first upper
    mat[3,end-1] = dt_dV*tmp3                   # last lower
    mat[2,end] = -dt_dV * ( tmp3*exp_vdV_D(v[end-1],dV,D)
                          +tmp2*(1.0 +exp_vdV_D(v[end],dV,D)) )  # last diagonal
    return

end;


function Integrate!(pop::SinglePopulation,μ::Float64, σ2::Float64,S::Vector{Float64},FS=true)

    #Unpack the state
    NV=pop.NV
    p=@view S[1:NV]
    νH=@view S[NV+4:end] # History of firning rate
    ν,νd,c=S[NV+1],S[NV+2],S[NV+3]

    #Consider to include Adt in the State
    Adt=zeros(3,NV)
    v=zeros(NV+1)


    # Number of realizations in refractory period
    IntRef=sum(νH[end-pop.nref+1:end])*pop.dt
    # calculate mass inside and outside the comp. domain
    IntP = sum(p)*pop.dV
    # normalize the probability distribution
    p*=(1.0-IntRef)/IntP

    # Fokker-Planck Drift
    GetF!(v,pop.V,μ,pop.τ,0.0,pop.NV+1)
    # Fokker-Planck Diffusion
    D = 0.5*σ2
    #Update Adt bandend matrix
    Diagonals!(Adt,pop.NV,v,D,pop.dV,pop.dt)
    #Reinjecton of flux in H:
    p[pop.ib] += νH[1]*(pop.dt/pop.dV)
    Adt *= -1
    Adt[2,:] += ones(pop.NV)

    # solve the linear system
    p[1:end].=LAPACK.gtsv!(Adt[3,1:end-1],Adt[2,:], Adt[1,2:end], p)

    if v[end] != 0.0
        ν = v[end]*((1.0+exp((-v[end]*pop.dV)/D))/(1.0-exp((-v[end]*pop.dV)/D)))*p[end]
    else
        ν = 2*D/pop.dV * p[end]
    end

    #Update firing rate history
    if FS
        νN=pois_rand(ν*pop.N*pop.dt)/(pop.N*pop.dt) # !!!Search for faster implementation
    else
        νN=ν
    end

    νH[1:end-1]=νH[2:end]
    νH[end]=νN 

    νd = νd + pop.dt *(νH[end-pop.nδ]- νd)/pop.τD
    c=   c  + pop.dt*( -c/pop.τC +ν)


    #Update
    S[1:NV]=p
    S[NV+1],S[NV+2],S[NV+3]=ν,νd,c
    return nothing
end;





## Load Net 
function LoadNet(fname,NV,α,dt)
    file=h5open(fname)
    Net=read(file)
    close(file)

    μx= Net["SNParam"]["JExt"][:,1].*Net["SNParam"]["NExt"][:,1].*Net["SNParam"]["NuExt"][:,1] +Net["SNParam"]["IExt"][:,1] # External mean current
    σ2x=(Net["SNParam"]["JExt"][:,1].^2).*(Net["SNParam"]["DeltaExt"][:,1].^2 .+1).*Net["SNParam"]["NExt"][:,1].*Net["SNParam"]["NuExt"][:,1] # External variance current
    δ=Net["SNParam"]["DMin"][:,1];
    τD=Net["SNParam"]["TauD"][:,1];
    θ=Net["SNParam"]["Theta"][:,1];
    H=Net["SNParam"]["H"][:,1];
    N=Int.(Net["SNParam"]["N"][:,1]);
    τ0=Net["SNParam"]["Tarp"][:,1];
    τm= Net["SNParam"]["Beta"][:,1];
    τc,αc,g=Net["SNParam"]["TauC"][:,1],Net["SNParam"]["AlphaC"][:,1],Net["SNParam"]["GC"][:,1];
    # Load Connectivity parameters
    c=Net["CParam"]["c"]
    J=Net["CParam"]["J"]
    ΔJ=Net["CParam"]["Delta"];
    ##

    #Initialize cortical populations
    Npop=length(N)
    cm=[DefineSinglePopulation(Dict("Vm"=> -α*θ[1],"θ"=>θ[1],"H"=>H[1],"NV"=>NV,
                                        "dt"=>dt,"τ"=>τm[1],"τ0"=>τ0[1],"τD"=>τD[1],"τC"=>τc[1],"δ"=>δ[1],"g"=>g[1],"N"=>N[1]))];
    for n=2:Npop
    push!(cm,DefineSinglePopulation(Dict("Vm"=> -α*θ[n],"θ"=>θ[n],"H"=>H[n],"NV"=>NV,
                "dt"=>dt,"τ"=>τm[n],"τ0"=>τ0[n],"τD"=>τD[n],"τC"=>τc[n],"δ"=>δ[n],"g"=>g[n],"N"=>N[1])))
    end

    S=Vector{Float64}[]
    for n=1:Npop
        S=push!(S,InitializeState(cm[n]))
    end


    A,B=zeros(size(c)),zeros(size(c))
    for i=1:size(c)[1]
        for j=1:size(c)[2]
            A[i,j]=c[i,j]*N[j]*J[i,j]
            B[i,j]=c[i,j]*N[j]*(J[i,j]^2)*(1+ΔJ[i,j]^2)
        end
    end
    A=sparse(A)
    B=sparse(B)

    return cm,S,μx,σ2x,Npop,A,B
    
end

function LoadRates(ratename,T0,Δt,dt)
    d=readdlm(ratename)
    r=d[:,2:end]
    Events=r.*(dt)
    k=Int(round(Δt/dt))
    steps=Int(round(T0/Δt))
    tss=range(0,stop=T0-Δt,step=Δt)
    rss=zeros(size(tss)[1],size(r)[2])
    Threads.@threads for n=1:steps-1
        for j=1:size(r)[2]
            rss[n,j]=sum(Events[1+n*k:(n+1)*k,j])/Δt
        end
    end
    return tss[1:end-1,:],rss[1:end-1,:]
end

function InitializeNet(netName,NV,α,dt)
    cm,S,μx,σ2x,Npop,A,B=FP.LoadNet(netName,NV,α,dt);
    function Computeμ_σ2(rd)
        μ=μx + A*rd
        σ2=σ2x + B*rd
        return μ,σ2
    end
    
    return cm,S,Npop,Computeμ_σ2
end



function InitializePerseusNet(modulesfile,connectfile,NV,dt,α) 
    d=readdlm(modulesfile)
    N=Int.(d[:,1])
    τm=d[:,6]/1000
    θ=d[:,7]
    H=d[:,8]
    τ0=d[:,9]/1000
    NExt=d[:,4]
    νExt=d[:,5]
    JExt=d[:,2]
    ΔExt=d[:,3]
    τc=d[:,11]/1000
    g=d[:,12]*1000
    μx= JExt.*NExt.*νExt  # External mean current
    σ2x=(JExt.^2).*(ΔExt.^2 .+1).*NExt.*νExt # External variance current

    d=readdlm(connectfile)
    Npop=size(N,1)
    c=zeros(Npop,Npop)
    J=zeros(Npop,Npop)
    ΔJ=zeros(Npop,Npop)
    δm=zeros(Npop,Npop)
    δM=zeros(Npop,Npop)

    PostSyn=Int.(d[:,1].+1)
    PretSyn=Int.(d[:,2].+1)

    for i=1:length(PostSyn)
        c[PostSyn[i],PretSyn[i]]=d[i,3]
        J[PostSyn[i],PretSyn[i]]=d[i,7]
        ΔJ[PostSyn[i],PretSyn[i]]=d[i,8]
        δm[PostSyn[i],PretSyn[i]]=d[i,4]/1000
        δM[PostSyn[i],PretSyn[i]]=d[i,5]/1000


    end

    TD=0.05
    τD= (δM.*TD - δm)./(TD - 1) - (δM - δm)/log(TD)


    A,B=zeros(size(c)),zeros(size(c))
    for i=1:size(c)[1]
        for j=1:size(c)[2]
            A[i,j]=c[i,j]*N[j]*J[i,j]
            B[i,j]=c[i,j]*N[j]*(J[i,j]^2)*(1+ΔJ[i,j]^2)
        end
    end
    A=sparse(A)
    B=sparse(B)
    cm=[DefineSinglePopulation(Dict("Vm"=> -α*θ[1],"θ"=>θ[1],"H"=>H[1],"NV"=>NV,
                                "dt"=>dt,"τ"=>τm[1],"τ0"=>τ0[1],"τD"=>τD[1,1],"τC"=>τc[1],"δ"=>δm[1],"g"=>g[1],"N"=>N[1]))];
    for n=2:Npop
        push!(cm,DefineSinglePopulation(Dict("Vm"=> -α*θ[n],"θ"=>θ[n],"H"=>H[n],"NV"=>NV,
                                        "dt"=>dt,"τ"=>τm[n],"τ0"=>τ0[n],"τD"=>τD[n,n],"τC"=>τc[n],"δ"=>δm[n,n],"g"=>g[n],"N"=>N[1])))
    end

    S=Vector{Float64}[]
    for n=1:Npop
        S=push!(S,InitializeState(cm[n]))
    end
    return cm,S,μx,σ2x,Npop,A,B
end

end

