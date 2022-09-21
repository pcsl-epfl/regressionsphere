## Regression of a Target Function on the Sphere

Code for the paper [*Learning sparse features can lead to overfitting in neural networks*](https://arxiv.org/abs/2206.12314) appeared at NeurIPS 2022. 
Details for running experiments can be found in `experiments.md`.

### Neural Network Training
Run ``main.py``

**Architecture.**
Train a one hidden-layer fully-connected neural network on the mean square error.
    
**Data-set.** *Data samples* `x` can be drawn from 
- the `d`-dimensional normal distribution, `args.dataset = normal`;
- uniform distribution inside the sphere, `uniform`;
- or uniform distribution on the spherical surface, `sphere`.

The *target* is either 
- the sample norm `||x||`, `args.target = norm`;
- or a Gaussian random field computed through an ~infinite-width teacher network (`args.target = teacher`) with `relu` or `abs` activation function to some power `a`.

**Algorithm.** *Full batch gradient descent* can be performed with 
- the *alpha-trick* by setting `args.alpha` larger (lazy) or smaller (feature) than one. 
- a *regularization* `args.l` on the l2 norm of the parameters (`args.reg = 'l2'`), on the path norm `||w1|| * |w2|` (`args.reg = 'l1'`) or on the l1 norm `|w2|` by fixing the first layer weights on the unit sphere `||w1|| = 1` (`args.reg = 'l1'` and `args.w1_norm1 = 1`).

Additionally, *conic* gradient descent [Chizat and Bach, 2018] can be performed by setting `args.conic_gd = 1`.


### Kernel Ridge Regression (KRR)
Run ``main_krr.py``

**Student kernel.** Analytical NTK of an infinite-width one-hidden-layer neural network.

**Data-set.** *see above.*

**Ridge.** The regularization or ridge parameter can be fixed by `args.l` (default: `0`).
