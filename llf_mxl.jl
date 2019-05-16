# Log-likelihood function
function loglik(param);

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

	c = trans(b,w,DR);
	v = zeros(NDRAWS,NALTMAX-1,NCSMAX,NP);
	# v = zeros(NDRAWS,NALTMAX,NCSMAX,NP);

	if NF > 0
		ff = reshape(f,1,1,NF,1);
		vf = reshape(sum(XF.*ff,dims=3),NALTMAX-1,NCSMAX,NP);
		# vf = reshape(sum(XF.*ff,dims=3),NALTMAX,NCSMAX,NP);
	else
		vf = zeros(NALTMAX-1,NCSMAX,NP);
		# vf = zeros(NALTMAX,NCSMAX,NP);
	end

	if NR > 0
		cc = reshape(c,1,1,NR,NP,NDRAWS);
		v = XR.*cc;
		v = reshape(sum(v,dims=3),NALTMAX-1,NCSMAX,NP,NDRAWS);
		# v = reshape(sum(v,dims=3),NALTMAX,NCSMAX,NP,NDRAWS);
		v = v .+ vf;
	else
		v = vf;
	end

	v = exp.(v);
	v[findall(isinf,v)] .= 10^20;
	v = v.*S;
	# yv = sum(v.* Y,dims=1);

	p = 1 ./(1 .+ sum(v,dims=1));
	# p = yv ./sum(v,dims=1);
	p = prod(p,dims=2);
	p = reshape(sum(p,dims=4)./NDRAWS,1,NP);
	p[findall(!isfinite,p)] .= 1;

	ll = - sum(log.(p));

	return(ll)
end

# Gradient function
function grf(param)

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
	gg = v.*p;
	gg = reshape(gg,NALTMAX-1,NCSMAX,1,NP,NDRAWS);

	if NF > 0;
		grf = -repeat(gg,1,1,NF,1,1).*XF;
		grf = reshape(sum(sum(grf,dims=1),dims=2),NF,NP,NDRAWS);
	else
		grf = [];
	end

	if NR > 0
		gg =  -repeat(gg,1,1,NR,1,1).*XR;
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

	# Back to prob
	p = reshape(p,NCSMAX,NP,NDRAWS);
	p = prod(p,dims=1);
	gr = [grf;grb;grw];

	# Gradient

	gr = gr.*p;
	
	p = sum(p,dims=3)./NDRAWS;
	p[findall(!isfinite,p)] .= 1;
	g = sum(gr,dims=3);
	g = g./NDRAWS;

	g = g./p;

	g = reshape(-sum(g,dims=2),NF+NR+NR);
	return(g)
end