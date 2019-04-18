# MultiLogit
Discrete Choice Models for Julia

MultiLogit is a collection of Julia functions to estimate Discrete Choice Models.

At the moment, we provide routines to estimate:
- Conditional Logit
- Mixed Logit with MLHS draws

I've tried to optimize routines to take advantage of Julia capabilities, such as broadcasting. The Mixed Logit code is based on the codes provided by Kenneth Train on his website (https://eml.berkeley.edu/~train/software.html)

Comments and contributions are welcome!
