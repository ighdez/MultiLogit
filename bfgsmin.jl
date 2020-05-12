using LinearAlgebra: inv, Matrix, I, norm, diag, diagind
using Calculus: hessian
using Distributions: Normal, cdf

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
function stepp(fx,x,ds;sigma=0.9,guess=1)

	# This line search algorithm was ported from 
	# Angel Luis Lopez code. More information available
	# in http://www.angelluislopez.net

	function stepper(fx,x,ds,alpha);
		xx = x' + alpha*ds; 
		fcc,dfcc = fx(xx;grad=true)
		dfcc = ds'*dfcc
		return (fcc,dfcc)
	end

	# Scalars
	ro = 1e-02;
	t1 = 9;
	t2 = 0.1;
	t3 = 0.5;

	flag = 0;
	maxit = 15;

	alpha = fill(NaN,maxit+1)
	alpha[1] = 0

	alpha[2] = guess;

	f0,df0 = stepper(fx,x,ds,alpha[1])

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
		
		# Evaluate function and search with alpha_i
		ff[i],dff[i] = stepper(fx,x,ds,alpha[i])
		
		if (ff[i] <= finf) & (ff[i] > 0)
			flag = 1
			break
		end
		
		# Check Wolfe conditions
		if (ff[i] > f0 + alpha[i]*ro*df0) | (ff[i] >= ff[i-1]) # Changed from (ff[i] >= f0)
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
			fa[k],dfa[k] = stepper(fx,x,ds,a[k])
			fb[k],dfb[k] = stepper(fx,x,ds,b[k])
			
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
			
			mincubic = (-c3+sqrt(abs(c3^2-3*c4*dfam[k])))/(3*c4);
			
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

			ff[k],dff[k] = stepper(fx,x,ds,alpha[k]);
			
			# Check Wolfe Conditions
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

# BFGSMin function for Mixed Logit
function bfgsmin_mxl(llf,x0; maxiter=1000,tol=1e-06,verbose=false);
	
	# Initialize
	x = x0;
	f_val,g0 = llf(x;grad=true);
	f_old = copy(f_val);

	H0 = Matrix{Float64}(I,size(x)[1],size(x)[1]);
	g_diff = Inf;
	lambda = 1
	convergence = -1;
	iter = 0
	
	if verbose
		println("\n","Optimizing: Initial F-Value: ", string(round(f_val;digits=2)))
	end
	# Start algorithm
	for it = 1:maxiter;
		lambda = 1;

		# Construct direction vector and relative gradient
		d = inv(-H0)*g0;
		m = d'*g0;
		# Select step to satisfy Wolfe Conditions
		lambda = stepp(llf,x',d;sigma=0.5,guess=lambda);

		# Construct the improvement and gradient improvement
		x = x + lambda*d[:,1];
		
		# Update Hessian
		f1,g1 = llf(x;grad=true);
		
		s0 = lambda*d;
		y0 = g1 - g0;
		
		H0 = H0 + (y0*y0')./(y0'*s0) - (H0*s0*s0'*H0)./(s0'*H0*s0);
		
		g0 = copy(g1);
		f_val = copy(f1);
		
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
	
	if verbose
		println("\n","Computing approximate Hessian")
	end
	
	h_final = numhess(llf,x);

	
	results = Dict("convergence" => convergence, "iterations" => iter, "max_f" => f_val, "par_max" => x, "hessian" => h_final);
	
	return(results);
end

# Present results from a bfgsmin_mxl object
# Thanks to Riccardo Scarpa for the contribution!
function summary_mxl(results,auxiliary_dict = Param_info)

	# Recover estimation parameters
	x = results["par_max"];
	h_final = results["hessian"];
    vcv = inv(h_final); # variance-covariance
    se = sqrt.(diag(vcv));     # st. errors
    ts = abs.(x) ./ se ; # t-values
    ps = ones(size(ts)) - cdf.(Normal(0,1),ts); # p-values

	# Arrange names and add zeros if coefficients distribute normal with mean zero
	NF = size(auxiliary_dict["namesF"],1);
	NR = size(auxiliary_dict["namesR"],1);
	XRdist = auxiliary_dict["distR"];
	namesX = [auxiliary_dict["namesF"];"b_".*auxiliary_dict["namesR"];"w_".*auxiliary_dict["namesR"]];

	f_x = x[1:NF];
	f_se = se[1:NF];
	f_t = ts[1:NF];
	f_p = ps[1:NF];

	if any(XRdist.==5)
		b_x = zeros(NR);
		b_x[XRdist.!=5] = x[(NF+1):(NF+sum(XRdist.!=5))];

		b_se = zeros(NR);
		b_se[XRdist.!=5] = se[(NF+1):(NF+sum(XRdist.!=5))];

		b_t = zeros(NR);
		b_t[XRdist.!=5] = ts[(NF+1):(NF+sum(XRdist.!=5))];

		b_p = zeros(NR);
		b_p[XRdist.!=5] = ps[(NF+1):(NF+sum(XRdist.!=5))];

		w_x = x[(NF+sum(XRdist.!=5)+1):end];
		w_se = se[(NF+sum(XRdist.!=5)+1):end];
		w_t = ts[(NF+sum(XRdist.!=5)+1):end];
		w_p = ps[(NF+sum(XRdist.!=5)+1):end];
	else
		b_x = x[(NF+1):(NF+NR)];
		b_se = se[(NF+1):(NF+NR)];
		b_t = ts[(NF+1):(NF+NR)];
		b_p = ps[(NF+1):(NF+NR)];

		w_x = x[(NF+NR+1):end];
		w_se = se[(NF+NR+1):end];
		w_t = ts[(NF+NR+1):end];
		w_p = ps[(NF+NR+1):end];
	end

	x = [f_x;b_x;w_x];
	se = [f_se;b_se;w_se];
	ts = [f_t;b_t;w_t];
	ps = [f_p;b_p;w_p];

	xs = hcat(namesX,x,se,ts,ps); # grouping for output
	xs = [["Variable" "Coefficient" "Std.err." "T" "P-value"];xs]

	return(xs)
end
