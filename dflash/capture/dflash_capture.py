"""Capture DFlash/DFlash2 drafter training inputs during real serving (vLLM V2 model runner).

Generic version of vllm-sly-radiance's radiance_dflash_capture.py. With DFLASH_CAPTURE_DIR set it
wraps DFlashSpeculator.propose (DFlash2Speculator inherits it) and records, per request, the
target's concatenated aux hidden states (what the drafter's `fc` reads, [tokens, n_aux x hidden])
and the token at each position: exactly what the proposer receives, so a drafter trained on them
sees the served target (this checkpoint, this kernel path). Hidden states are stored as e4m3 with
one fp16 scale per token. Rejected draft slots (trailing each request's segment) are dropped.

Differences to the radiance original:
- prompt trimming: positions below prompt_len - DFLASH_CAPTURE_PROMPT_KEEP (default 2048, the
  drafter's sliding window) are not stored, so long prompts cost disk only for the context the
  drafter can attend to while drafting the answer. The file records `prompt_len`; positions stay
  absolute and contiguous from the first kept one.
- the hook installs itself lazily when vLLM imports the speculator module (no vLLM import at
  interpreter start), via dflash_capture.pth.

Serve with prefix caching OFF (cached prefixes produce no hidden states) and TP=1.
Env: DFLASH_CAPTURE_DIR, DFLASH_CAPTURE_PROMPT_KEEP (2048), DFLASH_CAPTURE_MAX_TOKENS (8192 stored/request).
"""
import atexit
import importlib.abc
import os
import sys

DIR = os.environ.get("DFLASH_CAPTURE_DIR", "")
KEEP = int(os.environ.get("DFLASH_CAPTURE_PROMPT_KEEP", "2048"))
MAX_TOKENS = int(os.environ.get("DFLASH_CAPTURE_MAX_TOKENS", "8192"))
TARGET = "vllm.v1.worker.gpu.spec_decode.dflash.speculator"
_buf: dict = {}
_plen: dict = {}
_stats = {"reqs": 0, "tokens": 0, "err": 0}


def _flush(req_id):
    import torch
    parts = _buf.pop(req_id, None)
    plen = _plen.pop(req_id, -1)
    if not parts:
        return
    pos = torch.cat([p[0] for p in parts]); tok = torch.cat([p[1] for p in parts])
    hid = torch.cat([p[2] for p in parts]); sc = torch.cat([p[3] for p in parts])
    order = torch.argsort(pos)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(req_id))[:100]
    torch.save({"positions": pos[order], "tokens": tok[order], "hid_e4m3": hid[order], "scale": sc[order],
                "prompt_len": plen}, os.path.join(DIR, f"{safe}.pt"))
    _stats["reqs"] += 1; _stats["tokens"] += int(tok.numel())
    if _stats["reqs"] in (1, 10, 100, 500, 1000, 2000, 4000, 8000):
        sys.stderr.write(f"[dflash_capture] {_stats['reqs']} requests, {_stats['tokens']} tokens written to {DIR}\n")


def _flush_all():
    for r in list(_buf):
        _flush(r)


def _capture(self, input_batch, aux_hidden_states, num_rejected):
    import torch
    if not aux_hidden_states:
        return
    nr = int(input_batch.num_reqs)
    req_ids = list(input_batch.req_ids)[:nr]
    qsl = input_batch.query_start_loc_np
    plen = input_batch.prefill_len_np[:nr]
    n = int(input_batch.num_tokens)
    pos_all = self.target_input_buffers.positions[:n]
    pos_all = (pos_all[0] if pos_all.dim() == 2 else pos_all).to(torch.int32).cpu()
    rej = num_rejected[:nr].to(torch.int32).cpu().tolist()          # rejected draft slots trail each segment
    keep_rows = []
    segs = []
    for i, rid in enumerate(req_ids):
        a, b = int(qsl[i]), int(qsl[i + 1]) - int(rej[i])
        lo = int(plen[i]) - KEEP
        if b > a:
            p = pos_all[a:b]
            m = p >= lo
            idx = torch.nonzero(m).flatten() + a
            have = sum(x[1].numel() for x in _buf.get(rid, []))
            if idx.numel() and have < MAX_TOKENS:
                segs.append((rid, len(keep_rows), idx.numel(), int(plen[i])))
                keep_rows.append(idx)
    live = set(req_ids)
    if keep_rows:
        rows = torch.cat(keep_rows)
        rows_d = rows.to(aux_hidden_states[0].device)
        h = torch.cat([a_[:n].index_select(0, rows_d) for a_ in aux_hidden_states], dim=-1).float()
        scale = h.abs().amax(dim=-1).clamp_min(1e-12) / 448.0
        q = (h / scale[:, None]).to(torch.float8_e4m3fn).view(torch.uint8).cpu()
        sc = scale.to(torch.float16).cpu()
        tok = self.target_input_buffers.input_ids[:n].index_select(0, rows_d).to(torch.int32).cpu()
        pos = pos_all.index_select(0, rows)
        off = 0
        for rid, _, cnt, pl in segs:
            _buf.setdefault(rid, []).append((pos[off:off + cnt], tok[off:off + cnt], q[off:off + cnt], sc[off:off + cnt]))
            _plen[rid] = pl
            off += cnt
    for rid in [r for r in _buf if r not in live]:
        _flush(rid)


def install(mod):
    DFlashSpeculator = mod.DFlashSpeculator
    if getattr(DFlashSpeculator, "_dflash_capture", False):
        return
    os.makedirs(DIR, exist_ok=True)
    orig = DFlashSpeculator.propose

    def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
                num_sampled, num_rejected, *a, **kw):
        if not kw.get("dummy_run", False) and not kw.get("is_profile", False):
            try:
                _capture(self, input_batch, aux_hidden_states, num_rejected)
            except Exception as e:  # noqa: BLE001
                _stats["err"] += 1
                if _stats["err"] <= 3:
                    sys.stderr.write(f"[dflash_capture] capture failed: {e!r}\n")
        return orig(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
                    num_sampled, num_rejected, *a, **kw)

    DFlashSpeculator.propose = propose
    DFlashSpeculator._dflash_capture = True
    atexit.register(_flush_all)
    sys.stderr.write(f"[dflash_capture] ON -> {DIR} (e4m3 + per-token scale, prompt keep {KEEP}, max {MAX_TOKENS} tokens/request)\n")


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            import importlib.util
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_install(module):
            exec_module(module)
            try:
                install(module)
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"[dflash_capture] install failed: {e!r}\n")
        spec.loader.exec_module = exec_and_install
        return spec


if DIR and not any(isinstance(f, _Hook) for f in sys.meta_path):
    sys.meta_path.insert(0, _Hook())
