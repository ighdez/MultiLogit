#################################################
#												
# 			   BFGSMin - Julia v0.4			
#												
# Changelog:									
#												
# v0.4: -	For gradient-provided problems, selection
#			of step size by a line search algorithm based
#			in Fletcher (2000) section 2.6 and ported from 
#			Angel Luis Lopez code. More information available
#			in http://www.angelluislopez.net
#														
# v0.3: - 	Inclusion of improved hessian that
# 			uses broadcast capabilities of Julia,
#			based on the function 'Numerical_Hessian'
#			included in the R package 'CDM'.
#
#		-	Now it's possible to add a user-written
#			gradient function.
#
#		-	The non-improved gradient function was
#			deprecated.
#
#		-	A lot of bug fixes.
#
# v0.2: Inclusion of improved gradient for 		
#       optimization with high amount of data 	
#       or complicated LL Functions.
#
# v0.1: Initial version, at least operational.			
#												
#################################################

#######################################################################################################################
#
# Usage of BFGSMin:
#
# bfgsmin(f,x0,g=nothing; maxiter=1000,tol=1e-06,verbose=false,hess=false,difftype="central", diffeps=sqrt(eps()))
#
# Where:
#
# - f:		Objective function to be minimized.
# - x0:		Starting values.
# - g:		Gradient function. If it is not provided, BFGS will use a finite-difference approximation.
#
# Optional arguments:
# - maxiter:	Maximum number of iterations.
# - tol:		Relative gradient tolerance. BFGSMin will stop if |g(x)'(-H(x)^-1)g(x)| < tol.
# - verbose:	Print informative messages during optimization.
# - hess:		Should BFGSMin provide the approximate Hessian at optimum? 
#				If false, BFGSMin will return the BFGS approximation instead
#				of approximate Hessian.
# - difftype:	Difference type used in finite-difference gradient, BFGSMin can use a "forward"
#				difference (default) or a "central" difference.
# - diffeps:	Value for finite-difference step parameter. By default is sqrt(eps), where 'eps'
#				is the machine floating number precision.
#
#######################################################################################################################

using LinearAlgebra: inv, Matrix, I, norm, diagind
using Calculus: hessian

# Improved numerical gradient function (with broadcast operations)
function numgr(f,param; difftype="forward",ep=sqrt(eps()));
	K = size(param)[1];
	gr = fill(NaN,K)
	plus = fill(Float64[],K,1)
	minus = fill(Float64[],K,1)
	ej = Matrix{Float64}(I,K,K)*ep;
	
	# Generate step matrices
	for i = 1:K
		plus[i] = param .+ ej[i,:]
		minus[i] = param .- ej[i,:]
	end
	
	# Generate central or forward gradient
	if difftype == "central"
		gr = @. (f(plus) - f(minus))*0.5/ep
	elseif difftype == "forward"
		gr = (f.(plus) .- f(param))./ep
	else 
		error("Non-valid difference type in gradient")
	end
	return(gr);
end

# Improved numerical hessian function (with broadcast operations)
function numhess(f,param; ep=1e-05)
	# This code is based in the R function'Numerical_Hessian'
	# created by Alexander Robitzsch for the package 'CDM'
	# I acknowledge him and his team for their efforts.

	# Initialize
	K = size(param)[1];
	hs = fill(NaN,K,K);
	plus = fill(Float64[],K,1);
	plus2 = fill(Float64[],K,1);
	plusplus = fill(Float64[],K,K);
	ej = Matrix{Float64}(I,K,K)*ep;
	
	# Generate step matrices
	for i = 1:K;
		plus[i] = param .+ ej[i,:];
		plus2[i] = param .+ 2*ej[i,:];
		
		for j = i+1:K;
				plusplus[i,j] = param .+ ej[i,:] .+ ej[j,:];
				plusplus[j,i] = plusplus[i,j];
		end;
		plusplus[i,i] = zeros(K);
	end;
	
	# Generate relevant evaluations
	fplus = f.(plus);
	fplus2 = f.(plus2);
	fpp = f.(plusplus);
	fx = f(param);
	
	# Compute hessian by broadcasting
	hs = (fpp .- fplus  .- fplus' .+ fx)./(ep*ep);
	hs[diagind(hs)] = (fplus2 - 2*fplus .+ fx)./(ep*ep);

	return(hs)
end

# Function for step size computation
function stepp(fx,dfx,x,ds;sigma=0.1,guess=0.1)

function stepper(fx,dfx,x,ds,alpha);
	xx = x' + alpha*ds; 
	fcc = fx(xx)
	dfcc = ds'*dfx(xx)
	return (fcc,dfcc)
end

# Scalars
ro = 0.01;
t1 = 9;
t2 = 0.1;
t3 = 0.5;

flag = 0;
maxit = 15;

alpha = fill(NaN,maxit+1)
alpha[1] = 0

alpha[2] = guess;

f0,df0 = stepper(fx,dfx,x,ds,alpha[1])

ff = fill(NaN,maxit+1);
dff = fill(NaN,maxit+1);

ff[1] = f0; dff[1] = df0;
finf = 0;
u = (finf-f0)/(ro*df0);

# Bracketing
j = NaN
a = fill(NaN,maxit+1);
b = fill(NaN,maxit+1);
dfm = fill(NaN,maxit+1);
for i = 2:maxit

	j = copy(i)
	# f[i],df[i] = stepper(rf,rdf,x0,s0,alpha[i]);
	f1,df1 = stepper(fx,dfx,x,ds,alpha[i]);

	ff[j] = copy(f1); dff[j] = copy(df1);
	
	if ff[i] <= finf
		flag = 1
		break
	end

	if (ff[i] > f0 + alpha[i]*ro*df0) | (ff[i] >= f0)
		a[i] = alpha[i-1]; b[i] = alpha[i];
		break
	end
	
	if abs(dff[i]) <= -sigma*df0
		flag = 1
		break
	elseif dff[i] >= 0
		a[i] = alpha[i]; b[i] = alpha[i-1]
		break
	end
	
	if u <= 2*alpha[i] - alpha[i-1]
		alpha[i+1] = u;
	
	else
	
		# Interval
		
		intval = [2*alpha[i]-alpha[i-1], min(u,alpha[i]+t1*(alpha[i]-alpha[i-1]))];
		
		# Mapping on to [0,1]
		map = [alpha[i-1] alpha[i]]
		dfm[i-1] = (map[2]-map[1])*dff[i-1]
		dfm[i] = (map[2]-map[1])*dff[i]
		
		# Parameters of the Hermite interpolating cubic
		c1 = ff[i-1];
		c2 = dfm[i-1];
		c3 = 3*(ff[i]-ff[i-1]) - 2*dfm[i-1] - dfm[i];
		c4 = dfm[i-1] + dfm[i] - 2*(ff[i]-ff[i-1]);
		
		# Interval: alpha = a + z(b-a); alpha is in intval
		zmin = (intval[1]-map[1])/(map[2]-map[1]);
		zmax = (intval[2]-map[1])/(map[2]-map[1]);
		
		# min c(z) = c1+c2z+c3z^2+c4z^3,
		# where z belongs to [zmin,zmax]
		mincubic = (-c3+sqrt(abs(c3^2-3*c4*dfm[i-1])))/(3*c4);
		
		if mincubic<zmin
			mincubic=zmin
		elseif mincubic>zmax
			mincubic=zmax
		end
		
		z=[zmin;zmax;mincubic];
		
		c = c1.+c2.*z.+c3.*z.^2 .+c4.*z.^3;
		
		cmin,indc = findmin(c);
		z = z[indc]
		
		# New value for alpha
		alpha[i+1] = map[1] .+ z.*(map[2]-map[1])
	end
end

if flag > 0
	alpha = alpha[j]
	return(alpha)

	# Sectioning

else
	
	fa = fill(NaN,maxit+1)
	fb = fill(NaN,maxit+1)
	dfa = fill(NaN,maxit+1)
	dfb = fill(NaN,maxit+1)
	dfam = fill(NaN,maxit+1)
	dfbm = fill(NaN,maxit+1)
	
	for k = j:maxit
		# global alpha
		fa[k],dfa[k] = stepper(fx,dfx,x,ds,a[k])
		fb[k],dfb[k] = stepper(fx,dfx,x,ds,b[k])
		
		intval = [a[k]+t2*(b[k]-a[k]),b[k]-t3*(b[k]-a[k])];
		
		# Map to [0,1]
		map = [a[k] b[k]];
		
		dfam[k] = (map[2]-map[1])*dfa[k]
		dfbm[k] = (map[2]-map[1])*dfb[k]
		
		# Parameters of the Hermite interpolating cubic
		
        c1 = fa[k];
        c2 = dfam[k];
        c3 = 3*(fb[k]-fa[k]) - 2*dfam[k] - dfbm[k];
        c4 = dfam[k] + dfbm[k] - 2*(fb[k]-fa[k]);
		
		# interval: alpha = a + z(b-a); alpha is in intval
		zmin = (intval[1]-map[1])/(map[2]-map[1]);
		zmax = (intval[2]-map[1])/(map[2]-map[1]);
		
		# min c(z) = c1+c2z+c3z^2+c4z^3,
		# where z belongs to [zmin,zmax]
		
		mincubic = (-c3+sqrt(c3^2-3*c4*dfam[k]))/(3*c4);
		
		if mincubic<zmin
			mincubic = zmin
		elseif mincubic>zmax
			mincubic = zmax
		end
		
		z=[zmin;zmax;mincubic];
		
		c = c1.+c2.*z.+c3.*z.^2 .+c4.*z.^3;
		
		cmin,indc = findmin(c);
		z = z[indc]
		
		# New value for alpha
		alpha[k] = map[1] + z*(map[2]-map[1]);

		ff[k],dff[k] = stepper(fx,dfx,x,ds,alpha[k]);
		
		if (ff[k] > f0+ro*alpha[k]*df0) | (ff[k] >= fa[k])
			a[k+1] = a[k]
			b[k+1] = alpha[k]
		else
			if abs(dff[k]) <= -sigma*df0
				alpha=alpha[k]
				return(alpha)
			end
			
			a[k+1] = alpha[k]
			
			if (b[k]-a[k])*dff[k] >= 0
				b[k+1] = a[k]
			else
				b[k+1] = b[k]
			end
			
			if (a[k]-alpha[k])*dfa[k] <= sqrt(1e-30)
				println("Potential Round-off error, no progress is posible in the line search")
				return(alpha)
			end
		end
	end
end

return(alpha)
end

# BFGSMin function
function bfgsmin(f,x0,g=nothing; maxiter=1000,tol=1e-06,verbose=false,hess=false,difftype="central", diffeps=sqrt(eps()));
	
	# Initialize
	x = x0;
	f_val = f(x);
	f_old = copy(f_val);
	if isnothing(g)
		g0 = numgr(f,x; difftype=difftype,ep=diffeps);
	else
		g0 = g(x)
	end

	H0 = Matrix{Float64}(I,size(x)[1],size(x)[1]);
	g_diff = Inf;
	c1 = 1e-04;
	lambda = 1
	convergence = -1;
	iter = 0
	
	if verbose
		if isnothing(g)
			println("\n","Warning: BFGSMin will use a finite-difference gradient")
		end
		
		if !hess
			println("\n","Warning: BFGSMin will return the approximate BFGS Hessian update")
		end
		
		println("\n","Optimizing: Initial F-Value: ", string(round(f_val;digits=2)))
	end
	# Start algorithm
	for it = 1:maxiter;
		lambda = 1;

		# Construct direction vector and relative gradient
		d = inv(-H0)*g0;
		m = d'*g0;
		# Select step to satisfy the Armijo-Goldstein condition
		if isnothing(g)
			while true;
				x1 = x + lambda*d[:,1];
				
				f1 = try 
						f(x1)
					catch
						NaN
					end
				
				ftest = f_val + c1*lambda*m[1];

				if isfinite(f1) & (f1 <= ftest) & (f1>0);
					break
				else
					lambda = lambda./2;
				end
			end
		else
			lambda = stepp(f,g,x',d);
		end

		# Construct the improvement and gradient improvement
		x = x + lambda*d[:,1];
		
		# Update Hessian
		
		if isnothing(g)
			g1 = numgr(f,x; difftype=difftype,ep=diffeps);
		else
			g1 = g(x);
		end
		
		s0 = lambda*d;
		y0 = g1 - g0;
		
		H0 = H0 + (y0*y0')./(y0'*s0) - (H0*s0*s0'*H0)./(s0'*H0*s0);
		
		g0 = copy(g1);
		f_val = f(x);
		
		g_diff = abs(m[1]);
		
		# Check if relative gradient is less than tolerance
		if g_diff < tol;
			if verbose
				println("\n","Converged!")
			end
			convergence = 0;
			break
		end
		
		iter = iter + 1
		
		# Show information if verbose == true
		if verbose
			println("Optimizing: Iter No: ",string(iter)," / F-Value: ", string(round(f_val;digits=2))," / |g(x)'(-H(x)^-1)g(x)|: ",string(round(g_diff;digits=6))," / Step: ",string(round(lambda;digits=5)))
		end

	end
	
	if iter == maxiter;
		convergence = 2;
		println("\n","Maximum iterations reached reached. Convergence not achieved.")
	end
	
	if hess
		if verbose
			println("\n","Computing approximate Hessian")
		end
		
		h_final = numhess(f,x);
	else
		h_final = H0;
	end
	
	results = Dict("convergence" => convergence, "iterations" => iter, "max_f" => f_val, "par_max" => x, "hessian" => h_final);
	
	return(results);
end