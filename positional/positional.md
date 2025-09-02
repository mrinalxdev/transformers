# Positional Encoding

1. **Input Parameters**:

   - `seq_len` ($ T$): Length of the input sequence (number of tokens).
   - `d_model` ($ d$): Embedding dimension (must be even for sin/cos pairs).
   - `position` ($ pos$): Integer position of each token,$ pos \in \{0, 1, \dots, T-1\} $
   - `i`: Index of the dimension in the embedding, $i \in \{0, 1, \dots, d-1\}$.

2. **Positional Encoding Matrix**:

   - The output is a matrix $ PE \in \mathbb{R}^{T \times d} $, where each row corresponds to a position $ pos $ and each column corresponds to a dimension $ i $.

3. **Divisor Term**:

   - For each even index $ i $ (i.e., $ i = 0, 2, 4, \dots, d-2 $), compute:
     $$
     div_term_i = \exp\left( i \cdot \frac{-\ln(10000)}{d} \right) = \left( 10000 \right)^{-i/d}
     $$
   - This creates a geometric progression of frequencies, where \( 10000^{2i/d} \) controls the wavelength of oscillations.

4. **Sinusoidal Functions**:

   - For each position $ pos$ and dimension $i $:
     - If $ i $ is even ($ i = 2k $):
       $$
       PE(pos, 2k) = \sin\left( pos \cdot \left( 10000 \right)^{-2k/d} \right)
       $$
     - If \( i \) is odd (\( i = 2k+1 \)):
       $$
       PE(pos, 2k+1) = \cos\left( pos \cdot \left( 10000 \right)^{-2k/d} \right)
       $$
   - This uses the same frequency for each sin/cos pair (for $i = 2k$ and $i = 2k+1 $).

5. **Intuition**:

   - The encoding creates unique, continuous patterns for each position $ pos $ using sine and cosine functions.
   - Lower $ i $ (smaller $ k $) results in higher-frequency waves, while higher $ i $ results in lower-frequency waves.
   - The factor $ 10000^{-2k/d} $ ensures frequencies decrease geometrically, allowing the model to distinguish positions with varying granularity.

6. **Output**:
   - The matrix $ PE \in \mathbb{R}^{T \times d} $ has sinusoidal values where each row $ pos $ encodes the position using alternating sine and cosine functions with frequencies determined by $ i $.

This encoding ensures that positions are uniquely represented and relative distances between positions can be learned by the transformer.
