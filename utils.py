import jax.numpy as jnp


def mix_weights(beta):
    T = beta.shape[-1]
    batched = beta.shape[:-1]

    beta1m_cumprod = (1 - beta).cumprod(-1)
    assert beta1m_cumprod.shape == (*batched, T)

    res = jnp.pad(beta, (0, 1), mode='constant', constant_values=1) \
			* jnp.pad(beta1m_cumprod, (1, 0), mode='constant', constant_values=1)
    assert res.shape == (*batched, T+1)

    return res
