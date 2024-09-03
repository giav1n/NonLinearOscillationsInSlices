include("FP.jl")
using Plots,ProgressMeter,Statistics,DelimitedFiles
# Meta Parameters
NV=200; # Number of grid points
dt=1/10000  # [s];
α=0.5 # Vmin=-αθ
cm,S,μx,σ2x,Npop,A,B=FP.InitializePerseusNet("modules.ini","connectivity.ini",NV,dt,α);

Life=10 # [s]
steps=Int(round(Life/dt))
νD,νN=zeros(Npop),zeros(Npop);
c=zeros(Npop)
rateName="bifFP.dat"
rateFile=open(rateName,"w")
FS=false ## Finite size
I=zeros(Npop) 
I[1]=-μx[1]*0.
g=zeros(Npop)
g[1]=40
#Simulate
@showprogress for n=1:steps 
    μ=μx+A*νD +I
    σ2=σ2x +B*νD

    for i=1:Npop
        FP.Integrate!(cm[i],μ[i] -g[i]*c[i], σ2[i],S[i],FS)
        c[i]=S[i][cm[i].NV+3]
        νD[i]=S[i][cm[i].NV+2]
        νN[i]=S[i][end]
    end
    writedlm(rateFile ,[vcat(n*dt,νN)]) #save output on file
end
close(rateFile);

t,ν=FP.LoadRates(rateName,Life,0.001,dt)
plot(t,ν[:,1])
plot!(t,ν[:,3])


## Create Bigurcation Diagrams:
using Statistics
function Simulate(Ix,gx)
    State=0
    SW,LAS,HAS,BS=false,false,false,false
    cm,S,μx,σ2x,Npop,A,B=FP.InitializePerseusNet("modules.ini","connectivity.ini",NV,dt,α);

    Life=20 #[s]
    steps=Int(round(Life/dt))
    νD,νN=zeros(Npop),zeros(Npop);
    c=zeros(Npop)
    νE,νI=zeros(steps),zeros(steps)
    t=zeros(steps)
    FS=false ## Finite size
    I=zeros(Npop) 
    I[1]=Ix
    g=zeros(Npop)
    g[1]=gx
    #Simulate
    for n=1:steps 
        μ=μx+A*νD +I
        σ2=σ2x +B*νD

        for i=1:Npop
            FP.Integrate!(cm[i],μ[i] -g[i]*c[i], σ2[i],S[i],FS)
            c[i]=S[i][cm[i].NV+3]
            νD[i]=S[i][cm[i].NV+2]
            νN[i]=S[i][end]
        end

        t[n]=n*dt
        νE[n],νI[n]=νN[1],νN[3]
    end
    
    Indx=argmin(abs.(t.-10))
    t,νE,νI=t[Indx:end],νE[Indx:end],νI[Indx:end]

    if abs(mean(νE)-maximum(νE))>2
        SW=true
    elseif νE[end]<20
        LAS=true
        #println(νE[end],"  ",νI[end])
    else
        HAS=true
        #println(νE[end],"  ",νI[end])

    end

    #Check for Bistability
    if LAS
        Life=0.3 # [s]
        steps=Int(round(Life/dt))
        nJump=30
        Jumps=vcat(1:nJump,nJump-1:-1:0)
        #Staircase
        for k=1:length(Jumps)
            for n=1:steps 
                μ=μx+A*νD +I
                σ2=σ2x +B*νD
                μ[1]+=Jumps[k]*0.01*μx[1]
        
                for i=1:Npop
                    FP.Integrate!(cm[i],μ[i] -g[i]*c[i], σ2[i],S[i],FS)
                    c[i]=S[i][cm[i].NV+3]
                    νD[i]=S[i][cm[i].NV+2]
                    νN[i]=S[i][end]
                end
            end
            #println("νE: ",νN[1],"  νI: ",νN[3])
        end

        for n=1:steps 
            μ=μx+A*νD +I
            σ2=σ2x +B*νD    
            for i=1:Npop
                FP.Integrate!(cm[i],μ[i] -g[i]*c[i], σ2[i],S[i],FS)
                c[i]=S[i][cm[i].NV+3]
                νD[i]=S[i][cm[i].NV+2]
                νN[i]=S[i][end]
            end
        end

        #println("νE: ",νN[1],"  νI: ",νN[3])
        if νN[1]>20
            BS=true
        end

     
    end
    if SW
        return 1
    else
        if BS
            return 2
        elseif HAS
            return 3
        elseif LAS
            return 4
        end  
    end

    return SW,LAS,HAS,BS

end

##
Ix=range(-0.3,stop=0.3,step=0.01)
gs=range(0.0,stop=80,step=1)
Z=zeros(length(gs),length(Ix))
for k=1:length(gs)
    println(gs[k])
    Threads.@threads for j=1:length(Ix)
        Z[k,j]=Simulate(Ix[j]*μx[1],gs[k])
    end
end
##
using JLD
save("Z2.jld",Dict("g"=>gs,"I"=>Ix.*μx[1],"Z"=>Z))
##
theme(:default)
heatmap(Ix,gs,Z,cmap=cgrad(:Set3_4, 4, categorical = true))
savefig("Bif.pdf")
