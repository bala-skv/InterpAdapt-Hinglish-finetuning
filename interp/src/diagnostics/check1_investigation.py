from __future__ import annotations
import sys, json, torch
from pathlib import Path
repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "bala"))
sys.path.insert(0, str(repo / "phase0"))
from model_loading import load_base_model, LoadConfig
from stage1.patching import NUM_LAYERS
sys.path.insert(0, str(repo / "stage1"))
from stage1_sanity_checks import (
    _score, _score_from_logits, _make_ids, _contrast, _recovery,
    _SOURCE_PROMPT, _SOURCE_TARGET, _TARGET_PROMPT, _TARGET_TARGET,
)
@torch.no_grad()
def capture_components(model, ids, patch_pos):
    o_proj = {}
    down_proj = {}
    embed = None
    handles = []
    def make_o(li):
        def _h(_m, args): o_proj[li] = args[0][0, patch_pos].detach().to(torch.float16).cpu()
        return _h
    def make_d(li):
        def _h(_m, args): down_proj[li] = args[0][0, patch_pos].detach().to(torch.float16).cpu()
        return _h
    def make_e(_m, args, out):
        nonlocal embed
        embed = out[0, patch_pos].detach().to(torch.float16).cpu()
    handles.append(model.model.embed_tokens.register_forward_hook(make_e))
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_o(i)))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_d(i)))
    try:
        model(input_ids=ids, use_cache=False)
    finally:
        for h in handles: h.remove()
    return o_proj, down_proj, embed
class ComponentPatch:
    def __init__(self, model, patch_pos, o_proj=None, down_proj=None, embed=None):
        self.model = model
        self.patch_pos = patch_pos
        self.o_proj = o_proj
        self.down_proj = down_proj
        self.embed = embed
        self.handles = []
    def __enter__(self):
        if self.embed is not None:
            def _e(_m, args, out):
                h = out.clone()
                h[0, self.patch_pos, :] = self.embed.to(device=h.device, dtype=h.dtype)
                return h
            self.handles.append(self.model.model.embed_tokens.register_forward_hook(_e))
        for i, layer in enumerate(self.model.model.layers):
            if self.o_proj is not None:
                def make_o(src):
                    def _h(_m, args):
                        h = args[0].clone()
                        h[0, self.patch_pos, :] = src.to(device=h.device, dtype=h.dtype)
                        return (h,) + args[1:]
                    return _h
                self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_o(self.o_proj[i])))
            if self.down_proj is not None:
                def make_d(src):
                    def _h(_m, args):
                        h = args[0].clone()
                        h[0, self.patch_pos, :] = src.to(device=h.device, dtype=h.dtype)
                        return (h,) + args[1:]
                    return _h
                self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_d(self.down_proj[i])))
        return self
    def __exit__(self, *_):
        for h in self.handles: h.remove()
        self.handles.clear()
def main():
    if not torch.cuda.is_available(): return 1
    device = torch.device("cuda", 0)
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    cfg = LoadConfig(precision="fp16")
    model, tok = load_base_model(cfg)
    model.eval()
    src_ids, src_plen = _make_ids(tok, _SOURCE_PROMPT, _SOURCE_TARGET, device)
    en_ids_n, en_plen = _make_ids(tok, _TARGET_PROMPT, _SOURCE_TARGET, device)
    en_ids_f, _       = _make_ids(tok, _TARGET_PROMPT, _TARGET_TARGET, device)
    src_pos = src_plen - 1
    tgt_pos = en_plen - 1
    m_clean = _contrast(_score(model, src_ids, src_plen), _score(model, *_make_ids(tok, _SOURCE_PROMPT, _TARGET_TARGET, device)))
    m_corrupted = _contrast(_score(model, en_ids_n, en_plen), _score(model, en_ids_f, en_plen))
    print(f"Baselines: clean={m_clean:.4f}, corrupted={m_corrupted:.4f}\n")
    def _rec(lg_n, lg_f):
        return _recovery(m_clean, m_corrupted, _contrast(_score_from_logits(lg_n, en_ids_n, en_plen), _score_from_logits(lg_f, en_ids_f, en_plen)))
    print("Capturing clean components...")
    c_o, c_d, c_e = capture_components(model, src_ids, src_pos)
    print("\n[1] OProj Only (Attention) ...")
    with ComponentPatch(model, tgt_pos, o_proj=c_o):
        r1 = _rec(model(input_ids=en_ids_n, use_cache=False).logits, model(input_ids=en_ids_f, use_cache=False).logits)
    print(f"Recovery = {r1:.4f}")
    print("\n[2] OProj + DownProj (Attention + MLP) ...")
    with ComponentPatch(model, tgt_pos, o_proj=c_o, down_proj=c_d):
        r2 = _rec(model(input_ids=en_ids_n, use_cache=False).logits, model(input_ids=en_ids_f, use_cache=False).logits)
    print(f"Recovery = {r2:.4f}")
    print("\n[3] OProj + DownProj + Embed (Complete Casual Path) ...")
    with ComponentPatch(model, tgt_pos, o_proj=c_o, down_proj=c_d, embed=c_e):
        r3 = _rec(model(input_ids=en_ids_n, use_cache=False).logits, model(input_ids=en_ids_f, use_cache=False).logits)
    print(f"Recovery = {r3:.4f}")
if __name__ == "__main__":
    raise SystemExit(main())