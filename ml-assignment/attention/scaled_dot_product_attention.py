from typing import Tuple
import numpy as np


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    
    if Q.shape[-1] != K.shape[-1]:
        raise ValueError("Last dimension of Q and K must match (d_k).")

    d_k = Q.shape[-1]

    # (..., seq_len_q, seq_len_k)
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d_k)

    if mask is not None:
        # Ensure mask is boolean for readability
        mask_bool = mask.astype(bool)
        # Very negative value so that masked positions become ~0 after softmax
        scores = np.where(mask_bool, scores, -1e9)

    # Numerically stable softmax along the last axis
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    scores_sum = np.sum(scores_exp, axis=-1, keepdims=True)
    # Avoid division by zero in degenerate cases
    scores_sum = np.where(scores_sum == 0.0, 1.0, scores_sum)

    weights = scores_exp / scores_sum

    # (..., seq_len_q, d_v)
    output = np.matmul(weights, V)
    return output, weights


def _demo() -> None:
    
    np.set_printoptions(precision=3, suppress=True)

    # Example with a single "batch" and sequence length 3
    Q = np.array([[1.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0],
                  [1.0, 1.0, 0.0]])

    K = np.array([[1.0, 0.0, 1.0],
                  [1.0, 1.0, 0.0],
                  [0.0, 1.0, 1.0]])

    V = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]])

    # Reshape to (seq_len, d_k) -> (1, seq_len, d_k) to match the docstring
    Q_batched = Q[None, ...]
    K_batched = K[None, ...]
    V_batched = V[None, ...]

    output, weights = scaled_dot_product_attention(Q_batched, K_batched, V_batched)

    print("Attention weights (shape {}):".format(weights.shape))
    print(weights[0])
    print("\nAttention output (shape {}):".format(output.shape))
    print(output[0])


if __name__ == "__main__":
    _demo()