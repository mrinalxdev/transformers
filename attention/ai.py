import jax
import jax.numpy as jnp
from typing import Tuple, Optional
import numpy as np

# Configuration
class Config:
    d_model = 128  # Model dimension
    num_heads = 4  # Number of attention heads
    max_seq_len = 64  # Maximum sequence length
    dilations = [1, 2, 4, 8]  # Dilation rates for heads
    dropout_rate = 0.1
    dtype = jnp.float32

# Initialize parameters
def init_params(rng: jax.random.PRNGKey, config: Config) -> dict:
    def init_linear(shape):
        return jax.random.normal(rng, shape, dtype=config.dtype) / jnp.sqrt(shape[-1])
    
    params = {}
    params['qkv'] = init_linear((config.d_model, config.d_model * 3))  # Query, Key, Value projections
    params['out'] = init_linear((config.d_model, config.d_model))  # Output projection
    params['ffn1'] = init_linear((config.d_model, config.d_model * 4))
    params['ffn2'] = init_linear((config.d_model * 4, config.d_model))
    params['ln1'] = {'scale': jnp.ones(config.d_model), 'bias': jnp.zeros(config.d_model)}
    params['ln2'] = {'scale': jnp.ones(config.d_model), 'bias': jnp.zeros(config.d_model)}
    return params

# Layer normalization
def layer_norm(x: jnp.ndarray, params: dict, epsilon: float = 1e-6) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + epsilon)
    return x * params['scale'] + params['bias']

# Dilated attention for a single head
def dilated_attention(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, 
                    dilation: int, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    seq_len, d_k = query.shape[-2:]
    scores = jnp.matmul(query, key.transpose(-1, -2)) / jnp.sqrt(d_k)
    
    # Apply dilation by selecting every `dilation`-th position
    if dilation > 1:
        indices = jnp.arange(0, seq_len, dilation)
        scores = scores.at[..., indices].get()
        value = value.at[..., indices, :].get()
    
    if mask is not None:
        scores = scores + mask
    
    attn_weights = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(attn_weights, value)

# Multi-scale dilated attention
def multi_scale_attention(x: jnp.ndarray, params: dict, config: Config, 
                        mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    batch, seq_len, d_model = x.shape
    d_k = d_model // config.num_heads
    
    # Project to query, key, value
    qkv = jnp.matmul(x, params['qkv']).reshape(batch, seq_len, 3, config.num_heads, d_k)
    q, k, v = jnp.split(qkv, 3, axis=2)
    q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)
    
    # Apply dilated attention per head
    outputs = []
    for i in range(config.num_heads):
        dilation = config.dilations[i]
        q_head = q[:, :, i, :]
        k_head = k[:, :, i, :]
        v_head = v[:, :, i, :]
        attn_output = dilated_attention(q_head, k_head, v_head, dilation, mask)
        outputs.append(attn_output)
    
    # Concatenate heads and project
    output = jnp.concatenate(outputs, axis=-1)
    output = output.reshape(batch, seq_len, d_model)
    return jnp.matmul(output, params['out'])

# Feed-forward network
def ffn(x: jnp.ndarray, params: dict) -> jnp.ndarray:
    hidden = jax.nn.relu(jnp.matmul(x, params['ffn1']))
    return jnp.matmul(hidden, params['ffn2'])

# Transformer layer with multi-scale attention
def transformer_layer(x: jnp.ndarray, params: dict, config: Config, 
                    mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    # Attention block
    x_norm = layer_norm(x, params['ln1'])
    attn_output = multi_scale_attention(x_norm, params, config, mask)
    x = x + attn_output  # Residual connection
    
    # Feed-forward block
    x_norm = layer_norm(x, params['ln2'])
    ffn_output = ffn(x_norm, params)
    return x + ffn_output  # Residual connection

# Causal mask for autoregressive attention
def create_causal_mask(seq_len: int) -> jnp.ndarray:
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    return jnp.where(mask == 0, -1e9, 0)

# Forward pass for the model
@jax.jit
def forward(params: dict, x: jnp.ndarray, config: Config, 
           mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    return transformer_layer(x, params, config, mask)

# Example usage
def main():
    config = Config()
    rng = jax.random.PRNGKey(0)
    params = init_params(rng, config)
    
    # Dummy input: batch_size=2, seq_len=64, d_model=128
    x = jax.random.normal(rng, (2, config.max_seq_len, config.d_model))
    mask = create_causal_mask(config.max_seq_len)
    
    # Forward pass
    output = forward(params, x, config, mask)
    
    # Compute loss (dummy example, assumes random targets)
    targets = jax.random.normal(rng, output.shape)
    loss = jnp.mean((output - targets) ** 2)
    
    # Compute gradients
    grad_fn = jax.grad(lambda p, x: jnp.mean((forward(p, x, config, mask) - targets) ** 2))
    grads = grad_fn(params, x)
    
    return output, loss, grads

# Run the example
if __name__ == "__main__":
    output, loss, grads = main()
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss}")

# Gradient update (simplified)
def update_params(params: dict, grads: dict, lr: float = 0.001) -> dict:
    updated_params = {}
    for key in params:
        if isinstance(params[key], dict):
            updated_params[key] = update_params(params[key], grads[key], lr)
        else:
            updated_params[key] = params[key] - lr * grads[key]
    return updated_params