import jax 
import jax.numpy as jnp
from typing import Tuple, Optional
import numpy as np


class Config:
    d_model = 128
    num_heads = 4
    max_seq_len = 64
    dilation = [1, 2, 4, 8]
    dropout_rate = 0.1
    dtype = jnp.float32



def init_params(rng: jax.random.PRNGKey, config : Config) -> dict:
    def init_linear(shape):
        return jax.random.normal(rng, shape, dtype=config.dtype) / jnp.sqrt(shape[-1])
    
    params = {}
    params['qkv'] = init_linear((config.d_model, config.d_model * 3))

    params['out'] = init_linear((config.d_model, config.d_model))
    params['ffn1'] = init_linear((config.d_model, config.d_model * 4))   
    params['ffn2'] = init_linear((config.d_model * 4, config.d_model))
    params['ln1'] = {'scale' : jnp.ones(config.d_model) , 'bias' : jnp.zeros(config.d_model)}
    params['ln2'] = {'scale' : jnp.ones(config.d_model), 'bias' : jnp.zeros(config.d_model)}


    return params

## Working with layer norm..
# I am loosing 10 hair strings for every 20 lines of code I write in Jax

def layer_norm(x : jnp.ndarray, params: dict, epsilon: float = 1e-6) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + epsilon)

    return x * params['scale'] + params['bias']




# dilated attention for a single head : Will work tomorrow GOOD NIGHT
# log : 4 sept 0320 hrs




