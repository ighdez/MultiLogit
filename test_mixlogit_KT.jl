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

# Some scalars
NP = 100;
NALT = 3
NCS = 1484;
NROWS = 1484*3;

# Declare variable that contains individuals
id = dat[:,1];

# Declare variable that contains every independent choice situation
csid = dat[:,2];

# Dependent variable
Ychoice = dat[:,3];

# Regressors with fixed parameters
Xfixed = [dat[:,10] dat[:,11]];

# Regressors with random parameters and their distributions
Xrandom = [ -dat[:,4]./10000 -dat[:,5] dat[:,6] dat[:,7] dat[:,9]];
XRdist = [2,2,2,1,1]; # 1 = Normal / 2: Log-normal

# Number of MLHS draws and fix random seed (I'm not sure if seed is working, please check!)
NDRAWS = 100;
Random.seed!(12439);

Bfixed = [0,0];
Brandom = [.2,.1,.1,-1.44,.412];
Wrandom = [0.01,0.01,0.01,0.01,0.01];

# Initialize parameters
param = [Bfixed;Brandom;Wrandom];

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

NR = size(Xrandom)[2];
NF = size(Xfixed)[2];

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
		# k = sum(cs.==t);
		S[1:k,1+t-t1,n] .= 1;
		# Y[1:k,1+t-t1,n] .= yy[(cs.==t)]
		if NF > 0
			XF[1:k,1+t-t1,:,n] = xxf[(cs.==t) .& (yy .== 0),:] .- repeat(xxf[(cs.==t) .& (yy .== 1),:],k,1);
			# XF[1:k,1+t-t1,:,n] = xxf[(cs.==t),:];
		end

		if NR > 0
			XR[1:k,1+t-t1,:,n] = xx[(cs.==t) .& (yy .== 0),:] .- repeat(xx[(cs.==t) .& (yy .== 1),:],k,1);
			# XR[1:k,1+t-t1,:,n] = xx[(cs.==t),:];
		end
	end
end

# Until here we created multidimensional arrays (makes computations easier)

# Create draws (only MLHS are available)
DR = makedraws();
DR = permutedims(DR,[3,2,1]);

# Uncomment to use the same draws as in KT code!
# using MATLAB
# DR = jarray(read_matfile("draws.mat")["DR"]);

# Call likelihood function and BFGS minimizer
include("llf_mxl.jl")
include("bfgsmin.jl")
include("misc.jl")

# Evaluate each involved function one time before iterate (makes computations faster)
loglik(param)
grf(param)
#numhess(loglik,param)

@time res = bfgsmin(loglik,param,grf;hess=true,verbose=true,tol=1e-05)
