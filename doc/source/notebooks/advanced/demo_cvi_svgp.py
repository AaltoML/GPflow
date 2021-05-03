
import itertools
import numpy as np
import time
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
from gpflow.ci_utils import ci_niter
from gpflow.optimizers import NaturalGradient

plt.style.use("ggplot")

# for reproducibility of this notebook:
rng = np.random.RandomState(123)
tf.random.set_seed(42)

# %% [markdown]
# ## Generating data
# For this notebook example, we generate 10,000 noisy observations from a test function:
# \begin{equation}
# f(x) = \sin(3\pi x) + 0.3\cos(9\pi x) + \frac{\sin(7 \pi x)}{2}
# \end{equation}


# Printing step
step = 1

# Parameters for optimisation
adam_lr = 0.01
natgrad_lr = 0.2
maxiter = 10


# %%
def func(x):
    return np.sin(x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)


N = 20  # Number of training observations
minibatch_size = N


X = rng.rand(N, 1) * 2 - 1  # X values
Y = func(X) + 0.2 * rng.randn(N, 1)  # Noisy Y values
data = (X, Y)

# %% [markdown]
# We plot the data along with the noiseless generating function:

# %%
plt.plot(X, Y, "x", alpha=0.2)
Xt = np.linspace(-1.1, 1.1, 1000)[:, None]
Yt = func(Xt)
_ = plt.plot(Xt, Yt, c="k")

# %% [markdown]
# ## Building the model
# The main idea behind SVGP is to approximate the true GP posterior with a GP conditioned on a small set of "inducing" values. This smaller set can be thought of as summarizing the larger dataset. For this example, we will select a set of 50 inducing locations that are initialized from the training dataset:

# %%
M = 10#N  # Number of inducing locations
#Z = np.linspace(X.min(), X.max(), M).reshape(-1, 1)
Z = X

m_cvi = gpflow.models.SVGP_CVI(
    gpflow.kernels.SquaredExponential(lengthscales=.1),
    gpflow.likelihoods.Gaussian(variance=.2**2), Z, num_data=N)

m_svgp = gpflow.models.SVGP(
    gpflow.kernels.SquaredExponential(lengthscales=.1),
    gpflow.likelihoods.Gaussian(variance=.2**2), Z, num_data=N, whiten=True)


def plot(m, title=""):
    plt.figure(figsize=(12, 4))
    plt.title(title)
    pX = np.linspace(-1, 1, 100)[:, None]  # Test locations
    pY, pYv = m.predict_y(pX)  # Predict Y values at test locations
    plt.plot(X, Y, "x", label="Training points", alpha=0.2)
    (line,) = plt.plot(pX, pY, lw=1.5, label="Mean of predictive posterior")
    col = line.get_color()
    plt.fill_between(
        pX[:, 0],
        (pY - 2 * pYv ** 0.5)[:, 0],
        (pY + 2 * pYv ** 0.5)[:, 0],
        color=col,
        alpha=0.6,
        lw=1.5,
    )
    Z = m.inducing_variable.Z.numpy()
    plt.plot(Z, np.zeros_like(Z), "k|", mew=2, label="Inducing locations")
    plt.legend(loc="lower right")



# %% [markdown]
# Now we can train our model. For optimizing the ELBO, we use the Adam Optimizer *(Kingma and Ba 2015)* which is designed for stochastic objective functions. We create a `run_adam` utility function  to perform the optimization.

# %%

# We turn off training for inducing point locations
for m in [m_cvi, m_svgp]:
    gpflow.set_trainable(m.inducing_variable, False)
    gpflow.set_trainable(m.likelihood, False)
    gpflow.set_trainable(m.kernel, False)


def run_optim_cvi(model, iterations):
    """
    Utility function running the Adam optimizer
    
    :param model: GPflow model
    :param interations: number of iterations
    """
    trainable_variables = model.kernel.trainable_variables + model.likelihood.trainable_variables

    # Create an Adam Optimizer action
    optimizer = tf.optimizers.Adam(lr=adam_lr)
    training_loss = model.training_loss_closure(data)
    logf = []

    #@tf.function
    def optimization_step():
        X, Y = data
        # natural gradient step

        model.natgrad_step(X, Y, lr=natgrad_lr)
        # hyper parameter step
        #optimizer.minimize(training_loss, var_list=trainable_variables)

    for s in range(iterations):
        print(s, -training_loss().numpy())
        optimization_step()
        if s % step == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
    return logf




def run_optim_svgp(model, iterations):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    trainable_variables = model.kernel.trainable_variables + model.likelihood.trainable_variables

    # Create an Adam Optimizer action
    optimizer = tf.optimizers.Adam(lr=adam_lr)
    training_loss = model.training_loss_closure(data)
    logf = []

    natgrad_opt = NaturalGradient(gamma=natgrad_lr)
    variational_params = [(model.q_mu, model.q_sqrt)]

    @tf.function
    def optimization_step():
        X, Y = data

        # natural gradient step
        natgrad_opt.minimize(training_loss, var_list=variational_params)
        # hyper parameter step
        #optimizer.minimize(training_loss, var_list=trainable_variables)

    for s in range(iterations):
        print(s, -training_loss().numpy())
        optimization_step()
        if s % step == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
    return logf



def run_optim_model(m, name):
    # first run (no opt)
    plt.figure()
    plot(m, "Predictions before training")
    plt.savefig('plots/%s_test_%d.svg' % (name,0))
    plt.close()

    it = 1
    if isinstance(m, gpflow.models.SVGP_CVI):
        run_optim = run_optim_cvi
    else:
        run_optim = run_optim_svgp

    logf = run_optim(m, maxiter)

    plt.figure()
    plt.plot(np.arange(maxiter)[::step], logf)
    plt.xlabel("iteration")
    _ = plt.ylabel("ELBO")
    plt.savefig('plots/%s_elbo_test_%d.svg'%(name,it))
    plt.close()

    # %% [markdown]
    # Finally, we plot the model's predictions.

    # %%
    plt.figure()
    plot(m, "Predictions after training")
    plt.savefig('plots/%s_test_%d.svg'%(name,it))
    plt.close()
    plt.show()


def run_optim_models(models, names):
    # first run (no opt)

    plt.figure()

    for m,name in zip(models, names):
        if isinstance(m, gpflow.models.SVGP_CVI):
            run_optim = run_optim_cvi
        else:
            run_optim = run_optim_svgp

        logf = run_optim(m, maxiter)

        plt.plot(np.arange(maxiter), logf, label=name)

    plt.xlabel("iteration")
    _ = plt.ylabel("ELBO")
    plt.legend()
    plt.savefig('plots/elbos_test.jpg')
    plt.close()


    for m,name in zip(models, names):
        plot(m,name)

    plt.show()


# run_optim_model(m_svgp, 'svgp')
# run_optim_model(m_cvi, 'cvi')

print(m_cvi.elbo(data))
print(m_svgp.elbo(data))


run_optim_models([m_cvi, m_svgp], ['cvi', 'svgp'])

