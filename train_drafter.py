"""Fine-tune the DFlash2 drafter on captured target data (self-distillation, DFlash recipe).

Data: radiance_dflash_capture files -- per request: tokens, positions, aux hidden states (e4m3 +
per-token scale, [n, 5*5120]). Model: DFlash2Qwen3 re-implemented in plain torch from vLLM's
qwen3_dflash2.py / qwen3_dflash.py (same math: fc -> hidden_norm -> per-layer K/V injection with
k_norm + RoPE, queries = [anchor token, 7 masks] with q/k norms + RoPE, non-causal block, sliding
window 2048, two-tap grouped convs around attention and MLP, final norm, target lm_head).
Loss (paper): CE at the 7 mask positions, weights exp(-(k-1)/gamma), gamma = 4 for block 8.
Frozen: target embed_tokens / lm_head, candidate selector. Trainable: fc, hidden_norm, layers, norm
(MLPs optional). Optimizer: AdamW with fp32 master weights + states on the CPU (ZeRO-offload style)
so the 1.8B-parameter drafter trains on one 32 GB card; GPU holds bf16 params + grads.
Eval: weighted CE and per-position top-1 accuracy on held-out sequences (acceptance proxy).

dflash-scripts version (from vllm-sly-radiance paroquant/drafter/train_drafter.py): reads captures from
capture/dflash_capture.py, whose positions start at prompt_len - KEEP (absolute RoPE positions are kept),
and by default places anchors only where the served drafter drafts: on the answer (anchor token at or
after prompt_len). --prompt-anchor-frac mixes in prompt anchors.
"""
import argparse, glob, json, math, os, random, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

p = argparse.ArgumentParser()
p.add_argument("--capture", required=True); p.add_argument("--drafter", required=True)
p.add_argument("--target", required=True, help="bf16 target dir (embed_tokens / lm_head)")
p.add_argument("--out", required=True)
p.add_argument("--epochs", type=float, default=2.0); p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--anchors", type=int, default=64, help="anchors per sequence per step")
p.add_argument("--seqs", type=int, default=4, help="sequences per step")
p.add_argument("--max-len", type=int, default=4096); p.add_argument("--min-ctx", type=int, default=32)
p.add_argument("--val", type=int, default=96); p.add_argument("--warmup", type=float, default=0.04)
p.add_argument("--val-dir", default="", help="fixed held-out capture dir (excluded from training); overrides --val")
p.add_argument("--gamma", type=float, default=4.0); p.add_argument("--freeze-mlp", action="store_true")
p.add_argument("--eval-only", action="store_true"); p.add_argument("--eval-every", type=int, default=150)
p.add_argument("--save-every", type=int, default=300); p.add_argument("--seed", type=int, default=0)
p.add_argument("--max-files", type=int, default=0); p.add_argument("--device", default="cuda")
p.add_argument("--hessians", default="", help="write GPTQ Hessians of the drafter's Linear inputs (query path) over the --val-dir captures to this file, then exit")
p.add_argument("--prompt-anchor-frac", type=float, default=0.0, help="share of anchors drawn from the prompt part")
args = p.parse_args()
torch.manual_seed(args.seed); random.seed(args.seed)
if args.device.startswith("cuda"):   # vllm-sly-radiance ships no flash / mem-efficient SDPA kernels for gfx1201 (hipErrorInvalidImage)
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
dev = torch.device(args.device)
cfg = json.load(open(os.path.join(args.drafter, "config.json")))
dc = cfg["dflash_config"]; H = cfg["hidden_size"]; NH = cfg["num_attention_heads"]; NKV = cfg["num_key_value_heads"]
HD = cfg["head_dim"]; L = cfg["num_hidden_layers"]; I = cfg["intermediate_size"]; EPS = cfg["rms_norm_eps"]
BLOCK = int(dc["block_size"]); MASK_ID = int(dc["mask_token_id"]); TAPS = int(dc["conv_kernel_size"]); GS = int(dc["conv_group_size"])
NAUX = len(dc["target_layer_ids"]); THETA = float(cfg["rope_parameters"]["rope_theta"]); WINDOW = int(cfg.get("sliding_window") or 0)
EMB_SCALE = float(dc.get("input_embedding_scale", 1.0)); NQ = BLOCK; NDRAFT = BLOCK - 1
assert cfg.get("is_causal") is False, "trainer assumes the non-causal DFlash2 block"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ model (plain torch)
def rms(x, w, eps=EPS):
    xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def rope_cos_sin(pos, dim=HD, theta=THETA):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=pos.device, dtype=torch.float32) / dim))
    f = pos.float()[:, None] * inv[None, :]                       # [n, dim/2]
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):                                      # x [n, heads, HD], neox (rotate half)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d].float(), x[..., d:].float()
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).to(x.dtype)


def grouped_conv(h, delta, base, block_size, groups, gs, taps):    # mirrors _grouped_conv
    blocks = h.unflatten(-1, (groups, gs))                         # [n, groups, gs]
    coef = base.view(1, taps, groups, gs) + delta.unsqueeze(-1)    # [n, taps, groups, gs]
    out = coef[:, 0] * blocks
    position = torch.arange(h.shape[0], device=h.device) & (block_size - 1)
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        out = out + coef[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return out.flatten(-2)


class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.groups = H // GS
        self.base_kernel = nn.Parameter(torch.zeros(2, TAPS, H, dtype=torch.bfloat16))
        self.kernel_projection = nn.Linear(H, 2 * TAPS * self.groups, bias=False, dtype=torch.bfloat16)

    def prepare(self, h):
        coef = self.kernel_projection(h).reshape(h.shape[0], 2, TAPS, self.groups)
        return grouped_conv(h, coef[:, 0], self.base_kernel[0], NQ, self.groups, GS, TAPS), coef[:, 1]

    def finish(self, h, coef):
        return grouped_conv(h, coef, self.base_kernel[1], NQ, self.groups, GS, TAPS)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        bf = dict(dtype=torch.bfloat16)
        self.input_layernorm = nn.Parameter(torch.ones(H, **bf)); self.post_attention_layernorm = nn.Parameter(torch.ones(H, **bf))
        self.q_proj = nn.Linear(H, NH * HD, bias=False, **bf); self.k_proj = nn.Linear(H, NKV * HD, bias=False, **bf)
        self.v_proj = nn.Linear(H, NKV * HD, bias=False, **bf); self.o_proj = nn.Linear(NH * HD, H, bias=False, **bf)
        self.q_norm = nn.Parameter(torch.ones(HD, **bf)); self.k_norm = nn.Parameter(torch.ones(HD, **bf))
        self.gate_proj = nn.Linear(H, I, bias=False, **bf); self.up_proj = nn.Linear(H, I, bias=False, **bf)
        self.down_proj = nn.Linear(I, H, bias=False, **bf)
        self.attention_conv = Conv(); self.mlp_conv = Conv()

    def context_kv(self, ctx_normed, cos, sin):                    # ctx_normed: hidden_norm(fc(aux)) [n, H]
        k = self.k_proj(ctx_normed).view(-1, NKV, HD); v = self.v_proj(ctx_normed).view(-1, NKV, HD)
        k = apply_rope(rms(k, self.k_norm), cos, sin)
        return k, v

    def forward(self, h, residual, qpos_cos, qpos_sin, ctx_k, ctx_v, mask):
        # h: [Q, H] queries (block-aligned, NQ per anchor); ctx_k/v: [n, NKV, HD]; mask: [Q, n + Q] bool (True = attend)
        if residual is None:
            residual = h; h = rms(h, self.input_layernorm)
        else:
            residual = residual + h; h = rms(residual, self.input_layernorm)
        h, coef = self.attention_conv.prepare(h)
        q = self.q_proj(h).view(-1, NH, HD); k = self.k_proj(h).view(-1, NKV, HD); v = self.v_proj(h).view(-1, NKV, HD)
        q = apply_rope(rms(q, self.q_norm), qpos_cos, qpos_sin); k = apply_rope(rms(k, self.k_norm), qpos_cos, qpos_sin)
        K = torch.cat([ctx_k, k], 0); V = torch.cat([ctx_v, v], 0)                 # [n+Q, NKV, HD]
        rep = NH // NKV
        K = K.repeat_interleave(rep, dim=1); V = V.repeat_interleave(rep, dim=1)    # [n+Q, NH, HD]
        qh = q.transpose(0, 1)[None]; kh = K.transpose(0, 1)[None]; vh = V.transpose(0, 1)[None]  # [1, NH, *, HD]
        att = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask[None, None], scale=HD ** -0.5)
        h = self.o_proj(att[0].transpose(0, 1).reshape(-1, NH * HD))
        h = self.attention_conv.finish(h, coef)
        residual = residual + h; h = rms(residual, self.post_attention_layernorm)
        h, coef = self.mlp_conv.prepare(h)
        h = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        h = self.mlp_conv.finish(h, coef)
        return h, residual


class Drafter(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(NAUX * H, H, bias=False, dtype=torch.bfloat16)
        self.hidden_norm = nn.Parameter(torch.ones(H, dtype=torch.bfloat16)); self.norm = nn.Parameter(torch.ones(H, dtype=torch.bfloat16))
        self.layers = nn.ModuleList([Layer() for _ in range(L)])

    def forward(self, aux, ctx_pos, q_ids, q_pos, mask, embed):
        ctx = rms(self.fc(aux), self.hidden_norm)
        ccos, csin = rope_cos_sin(ctx_pos); qcos, qsin = rope_cos_sin(q_pos)
        h = embed[q_ids] * EMB_SCALE if EMB_SCALE != 1.0 else embed[q_ids]
        residual = None
        for lyr in self.layers:
            ck, cv = lyr.context_kv(ctx, ccos, csin)
            h, residual = lyr(h, residual, qcos, qsin, ck, cv, mask)
        return rms(residual + h, self.norm)


def load_drafter(m, path):
    """Load z-lab's bf16 drafter (or dequantize tcclaviger's FP8 block-128 one)."""
    sd = {}
    with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
        keys = list(f.keys())
        for k in keys:
            if k.endswith("weight_scale_inv"): continue
            if k.endswith((".weight_scale", ".weight_shape")): continue
            if k.endswith(".weight_packed"):                      # compressed-tensors pack-quantized int4 g128 (syvai layout)
                from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
                base = k[: -len(".weight_packed")]
                shape = tuple(f.get_tensor(base + ".weight_shape").tolist())
                q = unpack_from_int32(f.get_tensor(k), 4, torch.Size(shape), packed_dim=1).float()
                sc = f.get_tensor(base + ".weight_scale").float()
                sd[base + ".weight"] = (q * sc.repeat_interleave(shape[1] // sc.shape[1], 1)).to(torch.bfloat16)
                continue
            t = f.get_tensor(k)
            if t.dtype == torch.float8_e4m3fn:
                s = f.get_tensor(k.replace(".weight", ".weight_scale_inv")).float()   # [N/128, K/128]
                N, K = t.shape
                s = s.repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)[:, :K]
                t = (t.float() * s).to(torch.bfloat16)
            sd[k] = t
    own = m.state_dict(); loaded = 0
    for k, v in sd.items():
        kk = k.replace("mlp.", "").replace("self_attn.", "") if k.startswith("layers.") else k
        if kk.startswith("candidate_selector."): continue
        if kk.endswith("_layernorm.weight") or kk.endswith("_norm.weight") or kk in ("hidden_norm.weight", "norm.weight"):
            kk = kk[: -len(".weight")]
        if kk not in own:
            raise KeyError(f"unmapped drafter tensor {k} -> {kk}")
        assert own[kk].shape == v.shape, (k, own[kk].shape, v.shape)
        own[kk].copy_(v.to(own[kk].dtype)); loaded += 1
    log(f"drafter: {loaded} tensors loaded from {path}")
    return {k: v for k, v in sd.items() if k.startswith("candidate_selector.")}


def load_target_embed_head(path):
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    # A text-only release (Qwen3_5ForCausalLM) names the embedding model.embed_tokens.
    emb = next(n for n in ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight") if n in idx)
    out = {}
    for name in (emb, "lm_head.weight"):
        with safe_open(os.path.join(path, idx[name]), "pt") as f:
            out[name] = f.get_tensor(name).to(torch.bfloat16)
    return out[emb], out["lm_head.weight"]


# ------------------------------------------------------------------ data
def load_seq(fn):
    """-> tokens, hid, scale, positions, lo, hi: anchors are drawn from [lo, hi) (row indices)."""
    d = torch.load(fn)
    pos = d["positions"].long(); n = pos.numel()
    if n < args.min_ctx + NQ + 1 or not bool((pos == pos[0] + torch.arange(n)).all()):
        return None
    n = min(n, args.max_len)
    plen = int(d.get("prompt_len", -1))
    hi = n - NQ - 1
    lo_prompt = min(args.min_ctx, hi - 1)
    lo = max(lo_prompt, plen - 1 - int(pos[0])) if plen > 0 else lo_prompt   # anchor token (row a+1) at position >= prompt_len
    if hi - lo < 2:
        return None
    return d["tokens"][:n].long(), d["hid_e4m3"][:n], d["scale"][:n], pos[:n], lo, hi, lo_prompt


def draw_anchors(lo, hi, lo_prompt, k, generator=None):
    if args.prompt_anchor_frac > 0 and lo - lo_prompt > 1:
        kp = int(round(k * args.prompt_anchor_frac))
        a = torch.cat([torch.randint(lo_prompt, lo, (kp,), generator=generator), torch.randint(lo, hi, (k - kp,), generator=generator)])
    else:
        a = torch.randint(lo, hi, (k,), generator=generator)
    return a.to(dev)


def dequant(hid_u8, scale):
    return (hid_u8.to(dev).view(torch.float8_e4m3fn).float() * scale.to(dev).float()[:, None]).to(torch.bfloat16)


def make_batch(tokens, n, anchors, pos0):
    """Anchor a: context = positions 0..a (hidden states known), anchor token t[a+1] at a+1, masks at
    a+2..a+BLOCK; targets t[a+2..a+BLOCK]. Returns q_ids, q_pos, targets [A, NDRAFT], mask [A*NQ, n + A*NQ]."""
    A = anchors.numel()
    off = torch.arange(NQ, device=dev)
    q_pos = (anchors[:, None] + 1 + off[None, :]) + pos0                          # [A, NQ] absolute positions
    q_ids = torch.full((A, NQ), MASK_ID, dtype=torch.long, device=dev)
    q_ids[:, 0] = tokens[anchors + 1]
    targets = tokens[(anchors[:, None] + 2 + off[None, :NDRAFT])]                 # [A, NDRAFT]
    ctx_pos = torch.arange(n, device=dev) + pos0
    qp = q_pos.reshape(-1)                                                          # [A*NQ]
    ctx_ok = ctx_pos[None, :] <= (anchors + pos0).repeat_interleave(NQ)[:, None]   # context up to the anchor
    if WINDOW:
        ctx_ok &= (qp[:, None] - ctx_pos[None, :]) < WINDOW
    blk = torch.arange(A, device=dev).repeat_interleave(NQ)
    blk_ok = blk[:, None] == blk[None, :]                                          # own block, non-causal
    mask = torch.cat([ctx_ok, blk_ok], 1)
    return q_ids.reshape(-1), qp, targets, mask


WEIGHTS = torch.exp(-(torch.arange(NDRAFT, dtype=torch.float32)) / args.gamma).to(dev)  # k = 1..NDRAFT -> exp(-(k-1)/gamma)


def seq_loss(model, embed, head, tokens, aux, anchors, pos0, want_acc=False):
    n = aux.shape[0]
    q_ids, q_pos, targets, mask = make_batch(tokens, n, anchors, pos0)
    hs = model(aux, torch.arange(n, device=dev) + pos0, q_ids, q_pos, mask, embed)  # [A*NQ, H]
    hs = hs.view(-1, NQ, H)[:, 1:, :]                                            # mask positions only
    logits = F.linear(hs.reshape(-1, H), head).float()                           # [A*NDRAFT, V]
    ce = F.cross_entropy(logits, targets.reshape(-1), reduction="none").view(-1, NDRAFT)
    loss = (ce * WEIGHTS[None, :]).sum(1).mean() / WEIGHTS.sum()
    acc = None
    if want_acc:
        am = logits.argmax(-1).view(-1, NDRAFT)
        acc = {"pos": (am == targets).float().mean(0),                            # per position
               "len_argmax": prefix_len(am, targets), "len_walk": prefix_len(selector_walk(hs, logits, q_ids.view(-1, NQ)[:, 0]), targets)}
    return loss, ce.mean(0).detach(), acc


def prefix_len(draft, targets):
    """mean number of leading draft tokens equal to the target's tokens (= accepted tokens per block)."""
    return (draft == targets).long().cumprod(1).sum(1).float().mean()


@torch.no_grad()
def selector_walk(hs, logits, anchor_ids):
    """DFlash2's served draft: top-K candidates per position, pairwise scores from the frozen candidate
    selector (vLLM qwen3_dflash2 _score_edges_reference), greedy walk (draft_sample_method greedy)."""
    if SEL is None:
        return logits.argmax(-1).view(-1, NDRAFT)
    K = SEL_TOPK
    A = hs.shape[0]
    unary, cand = logits.view(A, NDRAFT, -1).topk(K, dim=-1)                        # [A, L, K]
    hidden = F.linear(hs, SEL["hidden_projection"]).float()                          # [A, L, r]
    succ = SEL["successor_codebook"][cand].float()                                   # [A, L, K, r]
    prev_ids = torch.cat([anchor_ids[:, None, None].expand(-1, 1, K), cand[:, :-1]], 1)
    pred = SEL["predecessor_codebook"][prev_ids].float()                             # [A, L, K, r]
    scores = unary.float()[:, :, None] + torch.einsum("blpr,blcr->blpc", pred * hidden[:, :, None], succ)
    prev = torch.zeros(A, dtype=torch.long, device=hs.device); out = []
    for l in range(NDRAFT):
        idx = scores[torch.arange(A, device=hs.device), l, prev].argmax(-1)
        out.append(cand[torch.arange(A, device=hs.device), l, idx]); prev = idx
    return torch.stack(out, 1)


@torch.no_grad()
def evaluate(model, embed, head, files, tag):
    model.eval(); tot = 0.0; cnt = 0; ce_pos = torch.zeros(NDRAFT, device=dev); acc_pos = torch.zeros(NDRAFT, device=dev); la = 0.0; lw = 0.0
    g = torch.Generator(device="cpu").manual_seed(1234)
    for fn in files:
        s = load_seq(fn)
        if s is None: continue
        tokens, hid, sc, pos, lo, hi, lo_p = s; tokens = tokens.to(dev)
        anchors = draw_anchors(lo, hi, lo_p, min(args.anchors, 48), generator=g)
        loss, cep, acc = seq_loss(model, embed, head, tokens, dequant(hid, sc), anchors, int(pos[0]), want_acc=True)
        tot += loss.item(); ce_pos += cep; acc_pos += acc["pos"]; la += acc["len_argmax"].item(); lw += acc["len_walk"].item(); cnt += 1
    model.train()
    acc_pos /= max(cnt, 1); ce_pos /= max(cnt, 1)
    # expected accepted tokens per block if acceptance were exactly position-wise top-1 hits with prefix semantics
    exp_acc = 0.0; prob = 1.0
    for k in range(NDRAFT): prob *= acc_pos[k].item(); exp_acc += prob
    log(f"[eval {tag}] weighted CE {tot / max(cnt, 1):.4f} | top1 by position " + " ".join(f"{a:.3f}" for a in acc_pos.tolist())
        + f" | prefix-expected accepted/block {exp_acc:.3f} | measured accepted/block argmax {la / max(cnt, 1):.3f} walk {lw / max(cnt, 1):.3f} | {cnt} seqs")
    return tot / max(cnt, 1), exp_acc


# ------------------------------------------------------------------ main
files = sorted(f for d in args.capture.split(",") for f in glob.glob(os.path.join(d, "*.pt")) if os.path.getsize(f) > (args.min_ctx + 16) * 25700)  # drop warmup/short captures; comma-separated dirs
if args.max_files: files = files[: args.max_files]
random.Random(args.seed).shuffle(files)
if args.val_dir:
    val_files = sorted(f for f in glob.glob(os.path.join(args.val_dir, "*.pt")) if os.path.getsize(f) > (args.min_ctx + 16) * 25700)
    train_files = files
else:
    val_files, train_files = files[: args.val], files[args.val:]
log(f"{len(files)} capture files: {len(train_files)} train / {len(val_files)} val")
model = Drafter().to(dev)
selector_sd = load_drafter(model, args.drafter)
SEL = {k.split(".")[-2] if k.endswith(".weight") else k.split(".")[-1]: v.to(dev) for k, v in selector_sd.items()} or None
SEL_TOPK = int(dc.get("selector_top_k", 16))
embed, head = load_target_embed_head(args.target); embed = embed.to(dev); head = head.to(dev)
log(f"target embed {tuple(embed.shape)} / head {tuple(head.shape)} on {dev}")
if args.freeze_mlp:
    for lyr in model.layers:
        for m in (lyr.gate_proj, lyr.up_proj, lyr.down_proj):
            for q in m.parameters(): q.requires_grad_(False)
trainable = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
log(f"trainable params: {sum(q.numel() for _, q in trainable) / 1e9:.2f} B in {len(trainable)} tensors")

if args.hessians:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gptq_utils import accumulate_hessian
    stats = {}

    def hook(key):
        def pre(m, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).detach()
            st = stats.setdefault(key, {"H": torch.zeros(x.shape[-1], x.shape[-1], device=dev), "n": 0})
            st["H"], st["n"] = accumulate_hessian(st["H"], x, st["n"])
        return pre
    # q_proj sees the query-path input of q/k/v (k/v also project the context rows, which syvai measured
    # to be the worse calibration); gate_proj's input is gate/up's.
    model.fc.register_forward_pre_hook(hook("fc"))
    for i, lyr in enumerate(model.layers):
        lyr.q_proj.register_forward_pre_hook(hook(f"layers.{i}.self_attn.qkv_proj"))
        lyr.o_proj.register_forward_pre_hook(hook(f"layers.{i}.self_attn.o_proj"))
        lyr.gate_proj.register_forward_pre_hook(hook(f"layers.{i}.mlp.gate_up_proj"))
        lyr.down_proj.register_forward_pre_hook(hook(f"layers.{i}.mlp.down_proj"))
evaluate(model, embed, head, val_files, "before")
if args.hessians:
    torch.save({k: {"H": v["H"].cpu(), "n": v["n"]} for k, v in stats.items()}, args.hessians)
    log(f"hessians -> {args.hessians}: " + ", ".join(f"{k}={v['n']}" for k, v in list(stats.items())[:3]) + f" ... ({len(stats)} modules)")
    sys.exit(0)
if args.eval_only: sys.exit(0)

# CPU fp32 master + AdamW (offloaded optimizer); GPU keeps bf16 params/grads
def _pin(t): return t.pin_memory() if dev.type == "cuda" else t
master = [_pin(q.detach().float().cpu()).requires_grad_(True) for _, q in trainable]
opt = torch.optim.AdamW(master, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, foreach=True)
steps_per_epoch = math.ceil(len(train_files) / args.seqs); total = int(steps_per_epoch * args.epochs)
warm = max(1, int(total * args.warmup))
sched = lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
log(f"training: {total} steps ({args.seqs} seqs x {args.anchors} anchors each), lr {args.lr}, warmup {warm}")
step = 0; t0 = time.time(); ema = None
order = []
while len(order) < total * args.seqs:
    ep = train_files[:]; random.shuffle(ep); order += ep
os.makedirs(args.out, exist_ok=True)


def save(tag):
    sd = {}
    for k, v in model.state_dict().items():
        if k.startswith("layers."):
            parts = k.split(".")
            name = parts[2]
            if name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                parts.insert(2, "self_attn")
            elif name in ("gate_proj", "up_proj", "down_proj"):
                parts.insert(2, "mlp")
            k = ".".join(parts)
        if k.endswith("input_layernorm") or k.endswith("post_attention_layernorm") or k.endswith("q_norm") or k.endswith("k_norm") or k in ("hidden_norm", "norm"):
            k = k + ".weight"
        sd[k] = v.detach().to(torch.bfloat16).cpu().contiguous()
    sd.update({k: v.cpu().contiguous() for k, v in selector_sd.items()})
    save_file(sd, os.path.join(args.out, "model.safetensors"))
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    log(f"saved {len(sd)} tensors -> {args.out} ({tag})")


for step in range(total):
    lr = args.lr * sched(step)
    for g in opt.param_groups: g["lr"] = lr
    model.zero_grad(set_to_none=True)
    batch = order[step * args.seqs:(step + 1) * args.seqs]; used = 0; loss_acc = 0.0
    for fn in batch:
        s = load_seq(fn)
        if s is None: continue
        tokens, hid, sc, pos, lo, hi, lo_p = s; tokens = tokens.to(dev)
        anchors = draw_anchors(lo, hi, lo_p, args.anchors)
        loss, _, _ = seq_loss(model, embed, head, tokens, dequant(hid, sc), anchors, int(pos[0]))
        (loss / len(batch)).backward(); loss_acc += loss.item(); used += 1
    if used == 0: continue
    # grads -> CPU master, clip, step, params back
    gn2 = 0.0
    for (n_, q), m in zip(trainable, master):
        g = q.grad.detach().float().cpu() if q.grad is not None else torch.zeros_like(m)
        m.grad = g; gn2 += float(g.pow(2).sum())
    gn = math.sqrt(gn2); clip = min(1.0, 1.0 / (gn + 1e-6))
    if clip < 1.0:
        for m in master: m.grad.mul_(clip)
    opt.step()
    with torch.no_grad():
        for (n_, q), m in zip(trainable, master): q.copy_(m.to(torch.bfloat16), non_blocking=True)
    l = loss_acc / used; ema = l if ema is None else 0.95 * ema + 0.05 * l
    if step % 10 == 0 or step == total - 1:
        el = time.time() - t0
        log(f"step {step + 1}/{total} loss {l:.4f} ema {ema:.4f} gnorm {gn:.2f} lr {lr:.2e} | {el / (step + 1):.1f} s/step, eta {(total - step - 1) * el / (step + 1) / 60:.0f} min")
    if (step + 1) % args.eval_every == 0 and step + 1 < total:
        evaluate(model, embed, head, val_files, f"step {step + 1}")
    if (step + 1) % args.save_every == 0 and step + 1 < total:
        save(f"step {step + 1}")
evaluate(model, embed, head, val_files, "after")
save("final")
