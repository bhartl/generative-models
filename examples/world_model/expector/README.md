### Mixture Density Network
*[taken from [tonyduan/mdn](https://github.com/tonyduan/mdn) under the MIT-License]*

---

Lightweight implementation of a mixture density network [1] in PyTorch.

An MDN models the conditional distribution over a *scalar* response as a mixture of Gaussians.

![Mixture density model](img/17870bed581ed5d53c0b24e84ca488a6.svg "Mixture density model")

[comment]: <> (<p align="center"><img alt="$$&#10;p_\theta&#40;y|x&#41; = \sum_{k=1}^K \pi^{&#40;k&#41;} \mathcal{N}&#40;\mu^{&#40;k&#41;}, {\sigma^2}^{&#40;k&#41;}&#41;,&#10;$$" src="" align="middle" width="232.54644105pt" height="48.18280005pt"/></p>)

where the mixture distribution parameters are output by a neural network, trained to maximize overall log-likelihood. The set of mixture distribution parameters is the following.

![Mixture density model](img/89d606a285fc8c10fba5542b37dae2c4.svg "Mixture density model")

In order to predict the response as a *multivariate* Gaussian distribution (for example, in [2]), we assume a fully factored distribution (i.e. a diagonal covariance matrix) and predict each dimesion separately. Another possible approach would be to use an auto-regressive method like in [3], but we leave that implementation for future work.

#### Usage

```python
import torch
from mdn import MixtureDensityNetwork

x = torch.randn(5, 1)
y = torch.randn(5, 1)

# 1D input, 1D output, 3 mixture components
model = MixtureDensityNetwork(1, 1, 3)
pred_parameters = model(x)

# use this to backprop
loss = model.loss(x, y)

# use this to sample a trained model
samples = model.sample(x)
```

For further details see the `ex_1d` example in `mdn.py`, executable via:
```cmd
python mdn.py ex-1d -n <NUMBER OF ITERATIONS>
```

![ex_model](img/fig_1d.png "Example model output")



#### References

[1] Bishop, C. M. Mixture density networks. (1994).

[2] Ha, D. & Schmidhuber, J. World Models. *arXiv:1803.10122 [cs, stat]* (2018).

[3] Van Den Oord, A., Kalchbrenner, N. & Kavukcuoglu, K. Pixel Recurrent Neural Networks. in *Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48* 1747–1756.

#### License

This code is available under the MIT License.
