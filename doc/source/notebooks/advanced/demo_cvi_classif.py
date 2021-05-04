
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
var_gp = .6
len_gp = .5
# likelihood
var_noise = .1

m_cvi = CVI(data,
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Bernoulli())

m_vgp_white = gpflow.models.VGP(data,
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Bernoulli(), whiten=True)

m_vgp = gpflow.models.VGP(data,
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Bernoulli(), whiten=False)


print('pre-optim cvi :', m_cvi.elbo())
print('pre-optim vgp :', m_vgp.elbo())
print('pre-optim vgp_white :', m_vgp_white.elbo())


#=============================================== run natgrad

lr_natgrad = .5
nit = 20

# CVI
[m_cvi.update_variational_parameters(beta=lr_natgrad) for _ in range(nit)]
print('cvi :',  m_cvi.elbo())

natgrad_opt = NaturalGradient(gamma=lr_natgrad)
# VGP
variational_params = [(m_vgp.q_mu, m_vgp.q_sqrt)]
[natgrad_opt.minimize(m_vgp.training_loss, var_list=variational_params) for _ in range(nit)]
print('vgp :', m_vgp.elbo())

variational_params_white = [(m_vgp_white.q_mu, m_vgp_white.q_sqrt)]
[natgrad_opt.minimize(m_vgp_white.training_loss, var_list=variational_params_white) for _ in range(nit)]
print('vgp_white :', m_vgp_white.elbo())


#============================================================

set_trainable(m_vgp_white.kernel.lengthscales, False)
set_trainable(m_vgp.kernel.lengthscales, False)
set_trainable(m_cvi.kernel.lengthscales, False)


for m, s in zip([m_vgp, m_vgp_white],['', 'white']):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(m.kernel.trainable_variables)
        loss = m.elbo()
    de_dhyp_vgp = tape.gradient(loss, m.kernel.trainable_variables)

    with tf.GradientTape() as t2:
        t2.watch(m.kernel.trainable_variables)
        with tf.GradientTape() as t1:
            t1.watch(m.kernel.trainable_variables)
            loss = m.elbo()
        dy_dx = t1.gradient(loss, m.kernel.trainable_variables)
    d2y_dx2 = t2.gradient(dy_dx, m.kernel.trainable_variables)

    print('post-optim vgp %s grad :'%s, de_dhyp_vgp[0].numpy())
    print('post-optim vgp %s hess :'%s, d2y_dx2[0].numpy())



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
llh_vgp_white = np.zeros((N_grid,))
llh_cvi = np.zeros((N_grid,))
vars_gp = np.linspace(.05, 1., N_grid)

for i, v in enumerate(vars_gp):

    m_cvi.kernel.variance.assign(tf.constant(v))
#    m_cvi.kernel.lengthscales.assign(tf.constant(v))
    llh_cvi[i] = m_cvi.elbo().numpy()
    m_vgp.kernel.variance.assign(tf.constant(v))
#    m_vgp.kernel.lengthscales.assign(tf.constant(v))
    llh_vgp[i] = m_vgp.elbo().numpy()
    m_vgp_white.kernel.variance.assign(tf.constant(v))
#    m_vgp.kernel.lengthscales.assign(tf.constant(v))
    llh_vgp_white[i] = m_vgp_white.elbo().numpy()

plt.figure()

plt.plot(vars_gp, llh_cvi, label='cvi')
plt.plot(vars_gp, llh_vgp, label='vgp')
plt.plot(vars_gp, llh_vgp_white, label='vgp_white')
plt.vlines(var_gp, ymin=llh_cvi.min(), ymax=llh_cvi.max())
plt.ylim([llh_cvi.min()-.1 *(llh_cvi.max()-llh_cvi.min()), llh_cvi.max()+.1 *(llh_cvi.max()-llh_cvi.min())])
plt.legend()
plt.title('classification')
plt.savefig('llh_classif.png')
plt.close()


#=============================================== run SVGP

M = 3  # Number of inducing locations
Z = np.linspace(X.min(), X.max(), M).reshape(-1, 1)

m_scvi = gpflow.models.SVGP_CVI(
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Bernoulli(), Z, num_data=N)

m_svgp_white = gpflow.models.SVGP(
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Bernoulli(), Z, num_data=N, whiten=True)

m_svgp = gpflow.models.SVGP(
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Bernoulli(), Z, num_data=N, whiten=False)

set_trainable(m_svgp_white.kernel.lengthscales, False)
set_trainable(m_svgp.kernel.lengthscales, False)
set_trainable(m_cvi.kernel.lengthscales, False)


#=============================================== run natgrad

lr_natgrad = .5
nit = 20

data = (X,Y)

# CVI
[m_scvi.natgrad_step(X,Y,lr_natgrad) for _ in range(nit)]
print('cvi :',  m_scvi.elbo(data))

natgrad_opt = NaturalGradient(gamma=lr_natgrad)
training_loss = m_svgp.training_loss_closure(data)
training_loss_white = m_svgp_white.training_loss_closure(data)

# SVGP
variational_params = [(m_svgp.q_mu, m_svgp.q_sqrt)]
[natgrad_opt.minimize(training_loss, var_list=variational_params) for _ in range(nit)]
print('vgp :', -training_loss().numpy())

variational_params_white = [(m_svgp_white.q_mu, m_svgp_white.q_sqrt)]
[natgrad_opt.minimize(training_loss_white, var_list=variational_params_white) for _ in range(nit)]
print('vgp_white :', -training_loss_white().numpy())

#========================================

N_grid = 100
llh_svgp = np.zeros((N_grid,))
llh_svgp_white = np.zeros((N_grid,))
llh_scvi = np.zeros((N_grid,))
vars_gp = np.linspace(.05, 1., N_grid)

for i, v in enumerate(vars_gp):

    m_scvi.kernel.variance.assign(tf.constant(v))
#    m_cvi.kernel.lengthscales.assign(tf.constant(v))
    llh_scvi[i] = m_scvi.elbo(data).numpy()
    m_svgp.kernel.variance.assign(tf.constant(v))
#    m_vgp.kernel.lengthscales.assign(tf.constant(v))
    llh_svgp[i] = m_svgp.elbo(data).numpy()
    m_svgp_white.kernel.variance.assign(tf.constant(v))
#    m_vgp.kernel.lengthscales.assign(tf.constant(v))
    llh_svgp_white[i] = m_svgp_white.elbo(data).numpy()

plt.figure()

plt.plot(vars_gp, llh_scvi, label='cvi')
plt.plot(vars_gp, llh_svgp, label='vgp')
plt.plot(vars_gp, llh_svgp_white, label='vgp_white')
plt.vlines(var_gp, ymin=llh_scvi.min(), ymax=llh_scvi.max())
plt.ylim([llh_scvi.min()-.1 *(llh_scvi.max()-llh_scvi.min()), llh_scvi.max()+.1 *(llh_scvi.max()-llh_scvi.min())])
plt.legend()
plt.title('classification')
plt.savefig('llh_classif_s.png')
plt.close()