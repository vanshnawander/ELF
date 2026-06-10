# Parallel Decoding in Continuous Diffusion Language Models (ELF)

This document provides a detailed technical explanation of how **ELF** decodes all $L$ tokens (e.g., $L = 1000$ sequence positions) in **parallel** from a final continuous latent state $z_1$, contrasting this with the sequential token-by-token generation of autoregressive models like GPT.

---

## 1. Core Concept: Parallel vs. Sequential Decoding

In traditional language models (like GPT), generation is **autoregressive** (sequential). Generating $L$ tokens requires $L$ model iterations:

```
Step 1: Prefix -> Model -> Token 1
Step 2: Prefix + Token 1 -> Model -> Token 2
Step 3: Prefix + Token 1 + Token 2 -> Model -> Token 3
```

In continuous diffusion models (like **ELF**), generation is **non-autoregressive** (parallel). Generating a text of maximum length $L$ requires only one decoding step after the SDE/ODE integration:

```
Step 1: Noise (1000 tokens) -> Solver Steps -> Continuous Latent z_1 (1000 tokens)
Step 2: Continuous Latent z_1 -> Unembedding Head -> 1000 Token Logits in Parallel
Step 3: 1000 Token Logits -> Parallel Argmax -> 1000 Tokens Output
```

---

## 2. Shape Transformations & Mathematical Steps

The decoding process operates on a batch size $B$, maximum sequence length $L = 1000$, transformer hidden dimension $d_{\text{hidden}} = 768$, encoder embedding dimension $d_{\text{model}} = 512$, and vocabulary size $V = 32,100$.

### Step 1: Continuous Latent Trajectory ($t = 0 \to 1$)
The SDE/ODE solver refines noise into clean embeddings $z_1$ in the continuous space:
$$z_1 \in \mathbb{R}^{B \times L \times d_{\text{model}}} \quad \rightarrow \quad \text{Shape: } (1, 1000, 512)$$

### Step 2: Transformer Hidden Representation
The latent $z_1$ is processed by the transformer blocks, yielding hidden states $h$:
$$h = \text{TransformerBlocks}(z_1) \in \mathbb{R}^{B \times L \times d_{\text{hidden}}} \quad \rightarrow \quad \text{Shape: } (1, 1000, 768)$$

### Step 3: Factored MLP Projection
To map the hidden states back to vocabulary space efficiently, a factored MLP projects $h$ down to $d_{\text{model}}$ before applying vocabulary weights:
$$h_{\text{proj}} = \text{GELU}(h \cdot W_{\text{proj}} + b_{\text{proj}}) \in \mathbb{R}^{B \times L \times d_{\text{model}}} \quad \rightarrow \quad \text{Shape: } (1, 1000, 512)$$
where:
* $W_{\text{proj}} \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{model}}}$
* $b_{\text{proj}} \in \mathbb{R}^{d_{\text{model}}}$

### Step 4: Vocabulary Unembedding
The projected representation is multiplied by the unembedding weights to compute the logits for every position and vocabulary item simultaneously:
$$\text{logits} = h_{\text{proj}} \cdot W_{\text{unembed}} + b_{\text{unembed}} \in \mathbb{R}^{B \times L \times V} \quad \rightarrow \quad \text{Shape: } (1, 1000, 32100)$$
where:
* $W_{\text{unembed}} \in \mathbb{R}^{d_{\text{model}} \times V}$
* $b_{\text{unembed}} \in \mathbb{R}^{V}$

### Step 5: Parallel Argmax Selection
The final discrete token IDs are chosen by extracting the index of the maximum logit value across the vocabulary axis (`axis=-1`) for all 1000 positions in parallel:
$$\hat{y} = \text{argmax}(\text{logits}, \text{axis}=-1) \in \mathbb{N}^{B \times L} \quad \rightarrow \quad \text{Shape: } (1, 1000)$$

---

## 3. Concrete Example Trace

Suppose we want to generate a short sentence inside a 1000-token limit: `"Continuous diffusion is fast."` 

Our vocabulary mappings are:
* `0` = `[PAD]` (Padding token)
* `1` = `[EOS]` (End-of-Sequence token)
* `321` = `"Continuous"`
* `56` = `"diffusion"`
* `12` = `"is"`
* `88` = `"fast"`
* `5` = `"."`

### A. The Logits Matrix (Shape: `[1, 1000, 32100]`)
Each row corresponds to one of the 1000 sequence positions:

$$\text{logits} = \begin{bmatrix}
\text{pos } 0: & [ -1.2, & 0.4, & \dots, & 12.8, & \dots ] & \leftarrow \text{max value is } 12.8 \text{ at index } 321 \\
\text{pos } 1: & [ -3.1, & -0.2, & \dots, & 10.5, & \dots ] & \leftarrow \text{max value is } 10.5 \text{ at index } 56 \\
\text{pos } 2: & [ -2.0, & 0.1, & \dots, & 14.2, & \dots ] & \leftarrow \text{max value is } 14.2 \text{ at index } 12 \\
\text{pos } 3: & [ -0.5, & 0.9, & \dots, & 11.9, & \dots ] & \leftarrow \text{max value is } 11.9 \text{ at index } 88 \\
\text{pos } 4: & [ -1.8, & -2.2, & \dots, & 9.8, & \dots ] & \leftarrow \text{max value is } 9.8 \text{ at index } 5 \\
\text{pos } 5: & [ -5.4, & 8.7, & \dots, & -1.2, & \dots ] & \leftarrow \text{max value is } 8.7 \text{ at index } 1 \\
\text{pos } 6: & [ 9.1, & -1.4, & \dots, & -4.8, & \dots ] & \leftarrow \text{max value is } 9.1 \text{ at index } 0 \\
\vdots & & & \vdots & & & \\
\text{pos } 999: & [ 8.9, & -2.1, & \dots, & -5.0, & \dots ] & \leftarrow \text{max value is } 8.9 \text{ at index } 0 \\
\end{bmatrix}$$

### B. Argmax Extraction in Code
Running `jnp.argmax` along `axis=-1` extracts the highest-valued token index for all 1000 rows simultaneously:
```python
predicted_ids = jnp.argmax(logits, axis=-1)
# Result: array of shape (1, 1000)
# [[321, 56, 12, 88, 5, 1, 0, 0, 0, ..., 0]]
```

### C. Masking Post-EOS
To ensure variable-length, open-ended answers without padding noise, any token appearing after the first End-of-Sequence token (`[EOS]`, token ID `1`) is overwritten with a pad token (`0`):
```python
# Turns all indices after the first EOS index into 0
predicted_ids = mask_after_eos(predicted_ids, eos_token_id=1, pad_token_id=0)
# Result: [[321, 56, 12, 88, 5, 1, 0, 0, 0, ..., 0]]
```

### D. Tokenizer Decoding
The array of 1000 IDs is passed to the T5 tokenizer to reconstruct the text:
```python
text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
print(text)
# Output: "Continuous diffusion is fast."
```
Although we allocated space for 1000 tokens, the output string naturally terminates where the `<EOS>` was generated, mimicking the variable-length responses of autoregressive models.
