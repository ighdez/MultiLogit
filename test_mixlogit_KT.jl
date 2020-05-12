# Uncomment these lines below if you need to install required packages (only once!!)
# using Pkg
# Pkg.add("CSV")
# Pkg.add("SpecialFunctions")
# Pkg.add("Random")
# Pkg.add("LinearAlgebra")
# Pkg.add("Calculus")

using CSV: read
using SpecialFunctions
using Random

include("misc.jl")
include("bfgsmin.jl")

# Load data
dat = read("data.txt",delim=" ",header=0);

# Declare variable that contains individuals
id = dat[:,1];

# Declare variable that contains every independent choice situation
csid = dat[:,2];

# Dependent variable
Ychoice = dat[:,3];

# Regressors with fixed parameters and their names
Xfixed = [dat[:,10] dat[:,11]];
XFnames = ["hiperf","medhiperf"];

# Regressors with random parameters,their distributions and names
Xrandom = [ -dat[:,4]./10000 -dat[:,5] dat[:,6] dat[:,7] dat[:,9]];
XRdist = [2,2,2,1,1]; # 1 = Normal / 2: Log-normal / 5: Normal with mean 0 
XRnames = ["price","opcost","range","ev","hybrid"];

# Number of MLHS draws and fix random seed (I'm not sure if seed is working, please check!)
NDRAWS = 100;
Random.seed!(12439);

Bfixed = [0,0];							# set as [] if no fixed coefficients
Brandom = [.2,.1,.1,-1.44,.412];			# set as [] if no random coefs. Set as zero if XRdist = 5. Will be taken as zero by the code regardless of the value
Wrandom = [0.01,0.01,0.01,0.01,0.01];	# Must have same dimension of Brandom

#######################################################
######## Do not change the code from this line ########
#######################################################

# Some scalars
NP = size(unique(id),1)
# NALT = 3;
NCS = size(unique(csid),1);
# NROWS = 1484*3;

# Initialize parameters
param = [Bfixed;Brandom[XRdist.!=5];Wrandom];

# Create information dictionary (for final presentation of results)
Param_info = Dict("namesF" => XFnames, "namesR" => XRnames, "distR" => XRdist);

# Create variables of NALTMAX and NCSMAX
nn = zeros(NCS);
for n=1:NCS;
	nn[n] = sum(csid .== n);
end
NALTMAX = trunc(Int,maximum(nn));

nn = zeros(NP);
for n=1:NP;
	k = (id .==n);
	k = csid[k]
	nn[n] = 1+k[end]-k[1];
end

NCSMAX = trunc(Int,maximum(nn));

if size(Xrandom,1) > 0;
	NR = size(Xrandom)[2];
else
	NR = 0;
end

if size(Xfixed,2) > 0;
	NF = size(Xfixed)[2];
else
	NF = 0;
end

# Generate matrices as in Kenneth Train's code
XR=zeros(NALTMAX-1,NCSMAX,NR,NP);
XF=zeros(NALTMAX-1,NCSMAX,NF,NP);
S=zeros(NALTMAX-1,NCSMAX,NP);

# From here, the code is a modification of the original code of Kenneth Train, with some tweaks to take advantage of Julia's features

# Loop over people
for n = 1:NP
	cs = csid[id .== n];
	yy = Ychoice[id .== n];

	if NF > 0
		xxf = Xfixed[id .== n,:]
	end

	if NR > 0
		xx = Xrandom[id .== n,:]
	end

	t1 = cs[1];
	t2 = cs[end];

	for t = t1:t2;
		k = sum(cs.==t)-1;
		S[1:k,1+t-t1,n] .= 1;

		if NF > 0
			XF[1:k,1+t-t1,:,n] = xxf[(cs.==t) .& (yy .== 0),:] .- repeat(xxf[(cs.==t) .& (yy .== 1),:],k,1);
		end

		if NR > 0
			XR[1:k,1+t-t1,:,n] = xx[(cs.==t) .& (yy .== 0),:] .- repeat(xx[(cs.==t) .& (yy .== 1),:],k,1);
		end
	end
end

# Until here we created multidimensional arrays (makes computations easier)

# Create draws (only MLHS are available)
DR = makedraws();
DR = permutedims(DR,[3,2,1]);

# Call likelihood function and BFGS minimizer
include("llf_mxl.jl")
include("bfgsmin.jl")
include("misc.jl")

# Evaluate each involved function one time before iterate (makes computations faster)
loglik(param)

@time numhess(loglik,param);
# @time hessian(loglik,param);

@time res = bfgsmin_mxl(loglik,param;verbose=true,tol=1e-06);

# Present Results
summary_mxl(res)