# Log-likelihood function
function loglik(param;grad=false);

	if NF > 0
		f = param[1:NF];
	else
		f = [];
	end

	if NR > 0
		b = param[(NF+1):(NF+NR)];
		w = param[(NF+NR+1):end];
	else
		b = [];
		w = [];
	end

	p = zeros(NP)
	g = zeros(NF+NR+NR,NP)

	c = trans(b,w,DR);
	v = zeros(NDRAWS,NALTMAX-1,NCSMAX,NP);


	if NF > 0
		ff = reshape(f,1,1,NF,1);
		vf = reshape(sum(XF.*ff,dims=3),NALTMAX-1,NCSMAX,NP);
	else
		vf = zeros(NALTMAX-1,NCSMAX,NP);
	end

	if NR > 0
		cc = reshape(c,1,1,NR,NP,NDRAWS);
		v = XR.*cc;
		v = reshape(sum(v,dims=3),NALTMAX-1,NCSMAX,NP,NDRAWS);
		v = v .+ vf;
	else
		v = vf;
	end

	v = exp.(v);
	v[findall(isinf,v)] .= 10^20;
	v = v.*S;
	p = 1 ./(1 .+ sum(v,dims=1));
	
	# Gradient
	if grad
		gg = v.*p;
		gg = reshape(gg,NALTMAX-1,NCSMAX,1,NP,NDRAWS);

		if NF > 0;
			grf = -gg.*XF;
			grf = reshape(sum(sum(grf,dims=1),dims=2),NF,NP,NDRAWS);
		else
			grf = [];
		end

		if NR > 0
			gg =  -gg.*XR;
			grb,grw = der(b,w,DR);
			grb = reshape(grb,1,1,NR,NP,NDRAWS);
			grw = reshape(grw,1,1,NR,NP,NDRAWS);
			grb = gg.*grb;
			grw = gg.*grw;
			grb = reshape(sum(sum(grb,dims=1),dims=2),NR,NP,NDRAWS);
			grw = reshape(sum(sum(grw,dims=1),dims=2),NR,NP,NDRAWS);
		else
			grb = [];
			grw = [];
		end
	end
	
	# Back to prob
	p = reshape(p,NCSMAX,NP,NDRAWS);
	p = prod(p,dims=1);
	
	if grad
		gr = [grf;grb;grw];
	
		# Gradient
		gr = gr.*p;
		g = sum(gr,dims=3);
	end
	
	# Back to prob
	p = sum(p,dims=3)./NDRAWS;
	p[findall(!isfinite,p)] .= 1;
	
	if grad
		# Gradient
		g = g./NDRAWS;
		g = g./p;
	end
	
	# Log-likelihood and gradient
	ll = - sum(log.(p));
	
	if grad
		g = reshape(-sum(g,dims=2),NF+NR+NR);
		return(ll,g)
	else
		return(ll)
	end
end
