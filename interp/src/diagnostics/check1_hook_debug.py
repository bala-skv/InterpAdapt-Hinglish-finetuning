import sys, torch
from pathlib import Path
repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "bala"))
sys.path.insert(0, str(repo / "phase0"))
sys.path.insert(0, str(repo / "stage1"))
from model_loading import load_base_model, LoadConfig
from stage1_sanity_checks import _make_ids, _SOURCE_PROMPT, _SOURCE_TARGET, _TARGET_PROMPT, _TARGET_TARGET
from check1_investigation import capture_components, ComponentPatch
def main():
    device = torch.device("cuda", 0)
    cfg = LoadConfig(precision="fp16")
    model, tok = load_base_model(cfg)
    model.eval()
    src_ids, src_plen = _make_ids(tok, _SOURCE_PROMPT, _SOURCE_TARGET, device)
    en_ids_n, en_plen = _make_ids(tok, _TARGET_PROMPT, _SOURCE_TARGET, device)
    src_pos = src_plen - 1
    tgt_pos = en_plen - 1
    c_o, c_d, c_e = capture_components(model, src_ids, src_pos)
    cor_o, cor_d, cor_e = capture_components(model, en_ids_n, tgt_pos)
    print(f"Embedding difference at patch_pos: {torch.norm(c_e - cor_e).item():.4f}")
    print(f"Layer 27 down_proj input diff: {torch.norm(c_d[27] - cor_d[27]).item():.4f}")
    final_states = {}
    def norm_hook(_m, args, out):
        final_states['out'] = out[0, tgt_pos].detach().cpu()
    h = model.model.norm.register_forward_hook(norm_hook)
    with ComponentPatch(model, tgt_pos, o_proj=c_o):
        model(input_ids=en_ids_n, use_cache=False)
        out_o = final_states['out'].clone()
    with ComponentPatch(model, tgt_pos, o_proj=c_o, down_proj=c_d, embed=c_e):
        model(input_ids=en_ids_n, use_cache=False)
        out_all = final_states['out'].clone()
    model(input_ids=src_ids, use_cache=False)
    out_clean = final_states['out'].clone()
    print(f"Difference between OProj-only patch and ALL patch: {torch.norm(out_o - out_all).item():.4f}")
    print(f"Difference between ALL patch and CLEAN run: {torch.norm(out_all - out_clean).item():.4f}")
    h.remove()
if __name__ == "__main__":
    main()