"""Minimal GPTQ (Frantar et al.) for symmetric group-wise int quantization, plus Hessian accumulation.

Taken unchanged in substance from syv-ai/qwen38-27b-rtx3090 drafter/gptq_utils.py (Apache-2.0), which
built syvai/Qwen3.8-27B-DFlash2-W4A16. Output matches compressed-tensors pack-quantized: int8 codes in
[-2^(b-1), 2^(b-1)-1], per-group scales, groups along K, no act-order (no g_idx).
"""
import torch


def accumulate_hessian(H, X, n_seen):
    """H: [K,K] fp32 running (2/n) * X^T X ; X: [n,K]. Returns (H, n_seen) updated (GPTQ's scaling)."""
    X = X.float()
    n = X.shape[0]
    H *= n_seen / (n_seen + n)
    n_seen += n
    X = (2.0 / n_seen) ** 0.5 * X
    H += X.t() @ X
    return H, n_seen


@torch.no_grad()
def gptq_quantize(W, H, bits=4, group=128, blocksize=128, percdamp=0.01):
    """W: [N,K], H: [K,K] fp32 Hessian of the layer input. Returns q int8 [N,K], scale fp32 [N, K//group]."""
    dev = W.device
    W = W.float().clone()
    N, K = W.shape
    assert K % group == 0 and blocksize == group
    qmax = 2 ** (bits - 1) - 1
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H))
    H += torch.eye(K, device=dev) * damp
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
    scales = torch.zeros(N, K // group, device=dev)
    for i1 in range(0, K, blocksize):
        i2 = i1 + blocksize
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        scale = torch.clamp(W1.abs().amax(dim=1) / qmax, min=1e-8)   # from the error-compensated weights
        scales[:, i1 // group] = scale
        for i in range(blocksize):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    q_int = torch.round(Q / scales.repeat_interleave(group, dim=1)).clamp(-qmax - 1, qmax).to(torch.int8)
    return q_int, scales


@torch.no_grad()
def rtn_quantize(W, bits=4, group=128):
    W = W.float()
    N, K = W.shape
    qmax = 2 ** (bits - 1) - 1
    g = W.reshape(N, K // group, group)
    scale = torch.clamp(g.abs().amax(dim=-1) / qmax, min=1e-8)
    q = torch.clamp(torch.round(g / scale[..., None]), -qmax - 1, qmax).to(torch.int8).reshape(N, K)
    return q, scale


def dequant(q, scale, group=128):
    N, K = q.shape
    return (q.float().reshape(N, K // group, group) * scale[..., None]).reshape(N, K)
