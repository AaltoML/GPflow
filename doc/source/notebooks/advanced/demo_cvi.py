
import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
from gpflow.models.cvi import CVI
from gpflow.optimizers import NaturalGradient

plt.style.use("ggplot")

# for reproducibility of this notebook:
rng = np.random.RandomState(123)
tf.random.set_seed(42)

# %%
def func(x):
    return np.sin(x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)

N = 20  # Number of training observations
X = rng.rand(N, 1) * 2 - 1  # X values
Y = func(X) + 0.2 * rng.randn(N, 1)  # Noisy Y values
data = (X, Y)

plt.plot(X, Y, "x", alpha=0.2)
Xt = np.linspace(-1.1, 1.1, 1000)[:, None]
Yt = func(Xt)
_ = plt.plot(Xt, Yt, c="k")

#plt.show()


var_gp = .5
len_gp = .1
var_noise = .1
lr_natgrad = 1.


m_gpr = gpflow.models.GPR(data,
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    noise_variance=var_noise
)

m_cvi = CVI(data,
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Gaussian(variance=var_noise))

m_vgp = gpflow.models.VGP(data,
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Gaussian(variance=var_noise))

print('pre-optim cvi :', m_cvi.elbo())
print('pre-optim svgp :', m_vgp.elbo())

m_cvi.update_variational_parameters(beta=lr_natgrad)
print('cvi :',  m_cvi.elbo())


natgrad_opt = NaturalGradient(gamma=lr_natgrad)
variational_params = [(m_vgp.q_mu, m_vgp.q_sqrt)]

natgrad_opt.minimize(m_vgp.training_loss, var_list=variational_params)
print('svgp :', m_vgp.elbo())

print('gpr :', m_gpr.log_marginal_likelihood())


N_grid = 50
llh_gpr = np.zeros((N_grid,))
llh_vgp = np.zeros((N_grid,))
llh_cvi = np.zeros((N_grid,))
vars_gp = np.linspace(.1, 2., N_grid)

for i, v in enumerate(vars_gp):

    m_gpr.kernel.variance.assign(tf.constant(v))
    llh_gpr[i] = m_gpr.log_marginal_likelihood().numpy()
    m_cvi.kernel.variance.assign(tf.constant(v))
    llh_cvi[i] = m_cvi.elbo().numpy()
    m_vgp.kernel.variance.assign(tf.constant(v))
    llh_vgp[i] = m_vgp.elbo().numpy()


print(llh_vgp)
print(llh_gpr)
print(llh_cvi)
plt.figure()

plt.plot(vars_gp, llh_cvi, label='cvi')
plt.plot(vars_gp, llh_vgp, label='vgp')
plt.plot(vars_gp, llh_gpr, ls='--', label='gpr')
plt.vlines(var_gp, ymin=llh_gpr.min(), ymax=llh_gpr.max())
plt.ylim([llh_gpr.min(), llh_gpr.max()+.1 *(llh_gpr.max()-llh_gpr.min())])
plt.legend()
plt.savefig('llh_gpr.png')
plt.close()



