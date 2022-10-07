## Reproducing figures of the paper *Learning sparse features can lead to overfitting in neural networks*
L. Petrini, F. Cagnetta, E. Vanden-Eijnden, M. Wyart. [Arxiv: 2206.12314](https://arxiv.org/abs/2206.12314)

Experiments are run using [`grid`](https://github.com/mariogeiger/grid/tree/master/grid).

### Figure 4
Neural network learning a constant function on the sphere with small **L1 regularization**:
- `d = 2` (here convergence is very slow, conic gradient descent [Chizat and Bach, 2018] is employed to speed it up)
```
python -m grid /home/results/regressionsphere --n 32 "
grun python main.py --init_w1 unitary --w1_norm1 1 --reg l1 --pofx sphere --pte 1000 --savefreq 10000
" --l 1e-6 --d 2 --dataseed 0 1 2 3 4 --ptr 1024 512 256 128 64 32 16 8  --maxstep 1e7 --h:int 10000 --conic_gd 1 --init_w2:str '1e-10'
```
- `d = 3`
```
python -m grid /home/results/regressionsphere --n 32 "
grun python main.py --init_w2 zero --init_w1 unitary --pofx sphere --pte 1000 --savefreq 10000 --w1_norm1 1 --reg l1
" --l 1e-5 --d 3  --dataseed 0 1 2 3 4 --ptr 1024 512 256 128 64 32 16 8 --h:int 10000 --maxstep 1e6
```
- `d = 5`
```
python -m grid /home/results/regressionsphere --n 32 "
grun python main.py --init_w2 zero --init_w1 unitary --pofx sphere --pte 1000 --savefreq 10000 --w1_norm1 1 --reg l1
" --l 1e-4 --d 5  --dataseed 0 1 2 3 4 --ptr 1024 512 256 128 64 32 16 8 --h:int 10000 --maxstep 1e5
```

Kernel regression of the constant on the sphere
```
python -m grid /home/results/krrsphere --n 4 "
grun python main_krr.py --target norm --pofx sphere --pte 10000
" --ptr 1024 512 256 128 32 16 8 --l 0 --d 2 3 5 --dataseed 0 1 2 3 4 5 6 7 8 9
```

### Figure G.3

Neural network learning a constant function on the sphere with the **alpha-trick**:
- `d = 2`
```
python -m grid /home/results/regressionsphere_alphatrick --n 12 "
grun python main.py
" --ptr 1024 512 256 128 64 32 16 8 --l 0 --d 2 --h:int 10000 --alpha 1e-6 --maxstep 1e7 --dataseed 0 1 2 3 4
```
- `d = 3`
```
python -m grid /home/results/regressionsphere_alphatrick --n 12 "
grun python main.py
" --ptr 1024 512 256 128 64 32 16 8 --l 0 --d 3 --h:int 10000 --alpha 1e-25 --maxstep 1e7 --dataseed 0 1 2 3 4
```
- `d = 5`
```
python -m grid /home/results/regressionsphere_alphatrick --n 12 "
grun python main.py
" --ptr 1024 512 256 128 64 32 16 8 --l 0 --d 5 --h:int 10000 --alpha 1e-50 --maxstep 1e6 --dataseed 0 1 2 3 4
```

The functions for counting the number of atoms can be found in `.arch.counting_atoms.py`.

### Figure G.1

Learning a GRF on the sphere. The teacher is an ~infinite-width (`H = 1e7`) FCN with `|.|^a` activation function, `a` controls the GRF smoothness `\nu_t = a + 1/2`.

Neural network:
- `a = 1`
```
python -m grid /home/results/regressionsphere_absteacher --n 16 "
grun python main.py --init_w1 unitary --w1_norm1 1 --reg l1 --target teacher --pofx sphere --teacher_act abs
 --pte 10000 --savefreq 10000 --h 10000 --init_w2 zero
" --ptr 1024 512 256 128 64 32 16 8 --act_power 1 --l 1e-5 --lr 0.1 --d 5 --dataseed 0 1 2 3 4 5 --maxstep 1e7
```
- `a = 4`
```
python -m grid /home/results/regressionsphere_absteacher --n 16 "
grun python main.py --init_w1 unitary --w1_norm1 1 --reg l1 --target teacher --pofx sphere --teacher_act abs
 --pte 10000 --savefreq 10000 --h 10000 --init_w2 zero
" --ptr 1024 512 256 128 64 32 16 8 --act_power 4 --l 1e-5 --lr 0.3 --d 5 --dataseed 0 1 2 3 4 5 --maxstep 1e7
```

Kernel regression:
```
python -m grid /home/results/krrsphere_absteacher --n 4 "
grun python main_krr.py --pofx sphere --target teacher --teacher_act abs --pte 10000
" --ptr 8192 4096 2048 1024 512 256 128 32 16 8 --act_power 1 6 --l 0 --d 5 --dataseed 0 1 2 3 4 5 6 7 8 9
```

### Figure 1

Experiments regarding FCNs training on images are run using the gradient flow approximation introduced in [Geiger et al. 2020](https://arxiv.org/abs/1906.08034), the corresponding code can be found here [github.com/leonardopetrini/feature_lazy](https://github.com/leonardopetrini/feature_lazy).

```
python -m grid /home/results/fc_on_images --n 16 "grun python main.py
 --init_kernel 0 --final_kernel 0 --delta_kernel 0 --pte 5000
  --arch fc --act softplus --L 1 --act_beta 5 --max_wall 20000 --max_dout 0.1
    --loss_beta 20 --max_dgrad 1e-4 --stop_frac .99 --h 1000 --alpha 1e-6
   " --seed_trainset 0 1 2 3 4 5 6 7 8 9 --dataset:str 'mnist' 'fashion' 'cifar10' --seed_init 0 1 2 3 4 5 6 7 8 9 --ptr 10000 5000 2500 1250 512 256 128 64 32
```

SVC trainings are performed using `sklearn.svm.SVC`.

### Figure 6

Image deformations are computed following [Petrini et al. 2021](https://arxiv.org/abs/2105.02468), the code can be found at [github.com/pcsl-epfl/diffeomorphism](https://github.com/pcsl-epfl/diffeomorphism).

`stability.py` contains the functions to compute predictors *rotation* stability,
```python
RS = rotation_stability(predictor, x, angle=10)
```
and *relative deformation* stability:
```python
D, G = deformation_and_noise_stability(predictor, imgs)
R = D / G
```