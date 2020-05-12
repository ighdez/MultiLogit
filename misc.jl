# Function of random draws by MLHS (ported from Kenneth Train's code)
function makedraws()
	dr = zeros(NDRAWS,NP,NR);
	h=Array(0:(NDRAWS-1));
	h=h./NDRAWS;
	for j=1:NR
		for n=1:NP
			draws=h .+rand(1)./NDRAWS; # Shift: Different shift for each person
			rr=rand(NDRAWS);
			rrid = sortperm(rr);
			rr = rr[rrid];
			draws=draws[rrid]; # Shuffle
			draws= -sqrt(2).*erfcinv.(2 .*draws); # Take inverse cum normal
			dr[:,n,j]=draws;
		end
	end
	
	return(dr)
end

# Function to transform random parameters (to be improved some day)
function trans(b,w,dr);
	if NR > 0
		c = b .+ w.*dr;
		c[(XRdist .== 2),:,:] = exp.(c[(XRdist .== 2),:,:]);
	else
		c = [];
	end
	return(c)
end

# Function of derivative of random parameters (to be improved some day)
function der(b,w,dr)
	db = ones(NR,NP,NDRAWS);
	if maximum(XRdist) > 1
		c = b .+ w.*dr;
		db[XRdist .== 2,:,:] = exp.(c[XRdist .== 2,:,:]);
	end
	dw = db.*dr;
	return(db,dw)
end
