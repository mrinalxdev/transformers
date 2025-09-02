# Relative Multi Head Attention

Standard multi head attention computes attention scores between query, key and value vectors. The relative variant adds learnable embeddings that capture the relative distance between positions the sequence, helping the model generalize better to longer sequences.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V
$$

Multi head splits this in $h$ parallel heads, concatenates outputs and projects them.

relative attention modifies the score and computation by adding a relative term :

$$\text{scores}_{i,j} = \frac{Q_i \cdot K_j^T + Q_i \cdot R_{i-j}^T}{\sqrt{d_k}}$$

Where $ R_{i-j} $ is a relative position embedding for the offset $ i-j $.
