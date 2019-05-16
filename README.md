# MultiLogit
Discrete Choice Models for Julia

MultiLogit is a collection of Julia functions to estimate Discrete Choice Models.

At the moment, I provide routines to estimate a Mixed Logit with MLHS draws

I've tried to optimize routines to take advantage of Julia capabilities, such as broadcasting. The Mixed Logit code is actually a port of the codes provided by Kenneth Train on his website (https://eml.berkeley.edu/~train/software.html) with some tweaks to take advantage of Julia's capabilities

Comments and contributions are welcome!
