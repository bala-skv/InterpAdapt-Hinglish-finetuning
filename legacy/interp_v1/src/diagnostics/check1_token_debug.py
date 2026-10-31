import sys, torch
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "stage1"))

from model_loading import load_base_model, LoadConfig
from stage1_sanity_checks import _make_ids, _SOURCE_PROMPT, _SOURCE_TARGET, _TARGET_PROMPT

def main():
    device = torch.device("cuda", 0)
    cfg = LoadConfig(precision="fp16")
    model, tok = load_base_model(cfg)
    model.eval()

    src_ids, src_plen = _make_ids(tok, _SOURCE_PROMPT, _SOURCE_TARGET, device)
    en_ids_n, en_plen = _make_ids(tok, _TARGET_PROMPT, _SOURCE_TARGET, device)
    
    src_pos = src_plen - 1
    tgt_pos = en_plen - 1

    print(f"\n--- TOKEN DEBUG ---")
    print(f"src_pos={src_pos}, tgt_pos={tgt_pos}")
    print(f"Token at src_pos (Hindi): ID={src_ids[0, src_pos]} -> {repr(tok.decode(src_ids[0, src_pos]))}")
    print(f"Token at tgt_pos (English): ID={en_ids_n[0, tgt_pos]} -> {repr(tok.decode(en_ids_n[0, tgt_pos]))}")

if __name__ == "__main__":
    main()