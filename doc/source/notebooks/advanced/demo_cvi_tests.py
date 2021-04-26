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

#### Some tests to check CVI is behaving ######

# %%
def func(x):
    return np.sin(x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)


def my_tf_round(x, decimals = 2):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier

N = 20  # Number of training observations
X = rng.rand(N, 1) * 2 - 1  # X values
Y = func(X) + 0.2 * rng.randn(N, 1)  # Noisy Y values
data = (X, Y)


var_gp = 2
len_gp = 2
var_noise = 1
lr_natgrad = .1
lr_sgd = .9


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


# Testing both elbos are the same
assert my_tf_round(m_cvi.elbo()) == my_tf_round(m_vgp.elbo()), "ELBOs do not match"


m_cvi.update_variational_parameters(beta=1)

# Testing both CVI lr=1 gives GPR likelihood
assert my_tf_round(m_cvi.elbo()) == my_tf_round(m_gpr.maximum_log_likelihood_objective()), "GPR and CVI do not match"

#Checking ELBO evaluations Natgrad and CVI

m_cvi = CVI(data,
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Gaussian(variance=var_noise))

set_trainable(m_cvi.kernel.lengthscales, True)
set_trainable(m_cvi.kernel.variance, True)
set_trainable(m_cvi.likelihood.variance, True)

m_cvi.update_variational_parameters(beta=lr_natgrad)

set_trainable(m_vgp.q_mu, False)
set_trainable(m_vgp.q_sqrt, False)

set_trainable(m_vgp.kernel.lengthscales, True)
set_trainable(m_vgp.kernel.variance, True)
set_trainable(m_vgp.likelihood.variance, True)

natgrad_opt = NaturalGradient(gamma=lr_natgrad)
variational_params = [(m_vgp.q_mu, m_vgp.q_sqrt)]
natgrad_opt.minimize(m_vgp.training_loss, var_list=variational_params)


# Testing one step in natgrad is same as CVI
assert my_tf_round(m_cvi.elbo()) == my_tf_round(m_vgp.elbo()), "GPR and CVI after one step do not match"


##### CLASSIFICATION ######

# %%
def func(x):
    return np.sin(x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)

N = 100  # Number of training observations
X = rng.rand(N, 1) * 2 - 1  # X values
F = func(X) + 0.2 * rng.randn(N, 1)  # Noisy Y values
Y = (F>0).astype(float)
data = (X, Y)


# gp
var_gp = .7
len_gp = .9
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

natgrad_opt = NaturalGradient(gamma=lr_natgrad)
variational_params = [(m_vgp.q_mu, m_vgp.q_sqrt)]
natgrad_opt.minimize(m_vgp.training_loss, var_list=variational_params)

print(' svgp :', m_vgp.elbo())

m_cvi.update_variational_parameters(beta=lr_natgrad)
start_level = m_cvi.elbo().numpy()

print('cvi :', m_cvi.elbo())

set_trainable(m_vgp.q_mu, False)
set_trainable(m_vgp.q_sqrt, False)


# Change what you want to train
set_trainable(m_vgp.kernel.lengthscales, True)
set_trainable(m_vgp.kernel.variance, False)


set_trainable(m_cvi.kernel.lengthscales, True)
set_trainable(m_cvi.kernel.variance, False)

k = 10
vgp_M = np.zeros(k)
vgp_E = np.zeros(k+1)
vgp_E[0] = m_vgp.elbo().numpy()

sgd_vgp = tf.optimizers.SGD(lr_sgd)

for i in range(k):
    sgd_vgp.minimize(m_vgp.training_loss, var_list=m_vgp.trainable_variables)
    vgp_M[i] = m_vgp.elbo().numpy()
    natgrad_opt.minimize(m_vgp.training_loss, var_list=variational_params)
    vgp_E[i+1] = m_vgp.elbo().numpy()


#print(m_vgp.elbo())

cvi_M = np.zeros(k)
cvi_E = np.zeros(k+1)
cvi_E[0] = m_cvi.elbo().numpy()

sgd_cvi = tf.optimizers.SGD(lr_sgd)

for i in range(k):
    sgd_cvi.minimize(m_cvi.neg_elbo, var_list=m_cvi.trainable_variables)
    cvi_M[i] = m_cvi.elbo().numpy()
    m_cvi.update_variational_parameters(beta=lr_natgrad)
    cvi_E[i+1] = m_cvi.elbo().numpy()


# Creating variables to plot
EM_cvi = np.zeros(cvi_E.shape[0] + cvi_M.shape[0])
EM_cvi[::2] = cvi_E
EM_cvi[1::2] = cvi_M

Ecvi_only = np.zeros(cvi_E.shape[0] + cvi_M.shape[0])
Ecvi_only[::2] = cvi_E
Ecvi_only[1::2] = None

Mcvi_only = np.zeros(cvi_E.shape[0] + cvi_M.shape[0])
Mcvi_only[::2] = None
Mcvi_only[1::2] = cvi_M

EM_vgp = np.zeros(vgp_E.shape[0] + vgp_M.shape[0])
EM_vgp[::2] = vgp_E
EM_vgp[1::2] = vgp_M

Evgp_only = np.zeros(vgp_E.shape[0] + vgp_M.shape[0])
Evgp_only[::2] = vgp_E
Evgp_only[1::2] = None

Mvgp_only = np.zeros(vgp_E.shape[0] + vgp_M.shape[0])
Mvgp_only[::2] = None
Mvgp_only[1::2] = vgp_M


plt.figure()
plt.plot(EM_cvi,label='CVI-EM_steps')
plt.plot(EM_vgp,label='VGP-EM_steps')
plt.plot(Ecvi_only,marker='x',color='black',alpha=0.5,label='E-step')
plt.plot(Mcvi_only,marker='o',color='green',alpha=0.5,label='M-step')
plt.plot(Evgp_only,marker='x',color='black',alpha=0.5)
plt.plot(Mvgp_only,marker='o',color='green',alpha=0.5)


plt.legend()

#plt.show()


##### Using CVI hyperparameters in the VGP model ######

m_cvi = CVI(data,
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Bernoulli())

m_vgp = gpflow.models.VGP(data,
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Bernoulli())

set_trainable(m_vgp.q_mu, False)
set_trainable(m_vgp.q_sqrt, False)

# Change what you want to train
set_trainable(m_vgp.kernel.lengthscales, True)
set_trainable(m_vgp.kernel.variance, False)


set_trainable(m_cvi.kernel.lengthscales, True)
set_trainable(m_cvi.kernel.variance, False)

# Change what you want to train
set_trainable(m_vgp.kernel.lengthscales, False)
set_trainable(m_vgp.kernel.variance, False)

set_trainable(m_cvi.kernel.lengthscales, True)
set_trainable(m_cvi.kernel.variance, False)

natgrad_opt = NaturalGradient(gamma=lr_natgrad)
variational_params = [(m_vgp.q_mu, m_vgp.q_sqrt)]
sgd = tf.optimizers.SGD(lr_sgd)

vgp_hyp = np.zeros(k)
vgp_fake = np.zeros(k)
vgp_E = np.zeros(k)


for i in range(k):
    natgrad_opt.minimize(m_vgp.training_loss, var_list=variational_params)
    m_cvi.update_variational_parameters(beta=lr_natgrad)
    sgd.minimize(m_cvi.neg_elbo, var_list=m_cvi.trainable_variables)
    var_t = tf.constant(m_cvi.kernel.lengthscales)
    #print(m_cvi.trainable_variables[0].numpy())
    vgp_E[i] = m_vgp.elbo().numpy()

    # create fake model
    m_vgp_fake = gpflow.models.VGP(data,
                            gpflow.kernels.SquaredExponential(lengthscales=tf.constant(m_vgp.kernel.lengthscales), variance=var_gp),
                            gpflow.likelihoods.Bernoulli())

    set_trainable(m_vgp_fake.q_mu, False)
    set_trainable(m_vgp_fake.q_sqrt, False)

    set_trainable(m_vgp_fake.kernel.lengthscales, True)
    set_trainable(m_vgp_fake.kernel.variance, False)

    m_vgp_fake.q_mu = tf.constant(m_vgp.q_mu)
    m_vgp_fake.q_sqrt = tf.constant(m_vgp.q_sqrt)

    m_vgp.kernel.lengthscales = var_t

    #sgd.minimize(m_vgp.training_loss, var_list=m_vgp.trainable_variables)
    sgd.minimize(m_vgp_fake.training_loss, var_list=m_vgp_fake.trainable_variables)

    vgp_hyp[i] = m_vgp.elbo().numpy()
    vgp_fake[i] = m_vgp_fake.elbo().numpy()

    print('vgp: ',m_vgp.elbo().numpy())
    print('cvi: ',m_cvi.elbo().numpy())
    print('fake:',m_vgp_fake.elbo().numpy())

# Creating variables to plot
vgp_norm = np.zeros(vgp_hyp.shape[0] + vgp_E.shape[0])
vgp_f = np.zeros(vgp_hyp.shape[0] + vgp_E.shape[0])


vgp_norm[::2] = vgp_E
vgp_norm[1::2] = vgp_hyp

vgp_f[::2] = vgp_E
vgp_f[1::2] = vgp_fake



plt.figure()
plt.plot(vgp_norm,label='VGP with CVI hyps')
plt.plot(vgp_f,label='VGP M-steps with VGP hyps',marker='o',color='black',alpha=0.5)

plt.legend()
plt.show()
