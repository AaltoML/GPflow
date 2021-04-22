
import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
from gpflow.models.cvi import CVI
from gpflow.optimizers import NaturalGradient
from gpflow import set_trainable

plt.style.use("ggplot")

# for reproducibility of this notebook:
rng = np.random.RandomState(123)
tf.random.set_seed(42)

# %%
def func(x):
    return np.sin(x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)

N = 100  # Number of training observations
X = rng.rand(N, 1) * 2 - 1  # X values
F = func(X) + 0.2 * rng.randn(N, 1)  # Noisy Y values
Y = (F>0).astype(float)
data = (X, Y)

plt.figure()
plt.plot(X, Y, "x", alpha=0.2)
Xt = np.linspace(-1.1, 1.1, 1000)[:, None]
Yt = func(Xt)
_ = plt.plot(Xt, Yt, c="k")
plt.savefig('data.png')
plt.close()

#plt.show()

# gp
var_gp = .9
len_gp = .5
# likelihood
var_noise = .1

m_cvi = CVI(data,
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Bernoulli())

m_vgp = gpflow.models.VGP(data,
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Bernoulli())

print('pre-optim cvi :', m_cvi.elbo())
print('pre-optim svgp :', m_vgp.elbo())


#=============================================== run natgrad

lr_natgrad = .5
nit = 20

# CVI
[m_cvi.update_variational_parameters(beta=lr_natgrad) for _ in range(nit)]
print('cvi :',  m_cvi.elbo())

# VGP
natgrad_opt = NaturalGradient(gamma=lr_natgrad)
variational_params = [(m_vgp.q_mu, m_vgp.q_sqrt)]
[natgrad_opt.minimize(m_vgp.training_loss, var_list=variational_params) for _ in range(nit)]
print('svgp :', m_vgp.elbo())

#============================================================

set_trainable(m_vgp.kernel.lengthscales, False)
set_trainable(m_cvi.kernel.lengthscales, False)

with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(m_vgp.kernel.trainable_variables)
    loss = m_vgp.elbo()
de_dhyp_vgp = tape.gradient(loss, m_vgp.kernel.trainable_variables)

with tf.GradientTape() as t2:
    t2.watch(m_vgp.kernel.trainable_variables)
    with tf.GradientTape() as t1:
        t1.watch(m_vgp.kernel.trainable_variables)
        loss = m_vgp.elbo()
    dy_dx = t1.gradient(loss, m_vgp.kernel.trainable_variables)
d2y_dx2 = t2.gradient(dy_dx, m_vgp.kernel.trainable_variables)

print('post-optim vgp grad :', de_dhyp_vgp[0].numpy())
print('post-optim vgp hess :', d2y_dx2[0].numpy())

with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(m_cvi.kernel.trainable_variables)
    loss = m_cvi.elbo()
de_dhyp_cvi = tape.gradient(loss, m_cvi.kernel.trainable_variables)

with tf.GradientTape() as t2:
    t2.watch(m_cvi.kernel.trainable_variables)
    with tf.GradientTape() as t1:
        t1.watch(m_cvi.kernel.trainable_variables)
        loss = m_cvi.elbo()
    dy_dx = t1.gradient(loss, m_cvi.kernel.trainable_variables)
d2y_dx2 = t2.gradient(dy_dx, m_cvi.kernel.trainable_variables)


print('post-optim cvi grad :', de_dhyp_cvi[0].numpy())
print('post-optim cvi hess :', d2y_dx2[0].numpy())

#========================================

N_grid = 100
llh_vgp = np.zeros((N_grid,))
llh_cvi = np.zeros((N_grid,))
vars_gp = np.linspace(.05, 1., N_grid)

for i, v in enumerate(vars_gp):

    m_cvi.kernel.variance.assign(tf.constant(v))
#    m_cvi.kernel.lengthscales.assign(tf.constant(v))
    llh_cvi[i] = m_cvi.elbo().numpy()
    m_vgp.kernel.variance.assign(tf.constant(v))
#    m_vgp.kernel.lengthscales.assign(tf.constant(v))
    llh_vgp[i] = m_vgp.elbo().numpy()


plt.figure()

plt.plot(vars_gp, llh_cvi, label='cvi')
plt.plot(vars_gp, llh_vgp, label='vgp')
plt.vlines(var_gp, ymin=llh_cvi.min(), ymax=llh_cvi.max())
plt.ylim([llh_cvi.min()-.1 *(llh_cvi.max()-llh_cvi.min()), llh_cvi.max()+.1 *(llh_cvi.max()-llh_cvi.min())])
plt.legend()
plt.title('classification')
plt.savefig('llh_classif.png')
plt.close()



