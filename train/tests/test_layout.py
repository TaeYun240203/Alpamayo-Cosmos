"""Token sequence layout and counts of the MORAI processor (MODEL_PLAN 2.5: ~1,750-token prefix). Needs the
Qwen/Qwen3-VL-8B-Instruct tokenizer files (HF cache)."""
from types import SimpleNamespace

import numpy as np, pytest, torch

from morai_vla.processor import BEV, CAMERAS, collate_fn, get_preprocess_fn, get_processor

CFG = SimpleNamespace(vlm_name_or_path="Qwen/Qwen3-VL-8B-Instruct", traj_vocab_size=4000, tokens_per_history_traj=48, tokens_per_future_traj=128)
TIMES = np.array([-0.31, -0.2, -0.1, -0.004])


def sample(cot=None, lidar=True, nav="Turn left in 11m. Speed limit 60 km/h"):
    g = torch.Generator().manual_seed(0)
    s = dict(camera_frames=[torch.randint(0, 255, (4, 3, h, w), generator=g, dtype=torch.uint8) for _, w, h in CAMERAS],
             camera_times=np.tile(TIMES, (4, 1)), nav_text=nav, cot=cot, key="ep/1", lidar_dropped=not lidar,
             ego_history_xyz=torch.zeros(1, 16, 3), ego_history_rot=torch.eye(3).expand(1, 16, 3, 3).clone(),
             ego_future_xyz=torch.zeros(1, 64, 3), ego_future_rot=torch.eye(3).expand(1, 64, 3, 3).clone())
    if lidar:
        s["bev_frames"] = torch.randint(0, 255, (2, 3, BEV[2], BEV[1]), generator=g, dtype=torch.uint8); s["bev_times"] = np.array([-0.21, -0.03])
    return s


def tok():
    return get_processor(CFG).processor.tokenizer


def ids_of(td):
    return tok()(td["text"], return_tensors="pt")["input_ids"][0]


def count(ids, token):
    return int((ids == tok().convert_tokens_to_ids(token)).sum())


def test_image_tokens_per_camera_and_bev():
    td = get_preprocess_fn(CFG)(data=sample())
    per_image = (td["image_grid_thw"].prod(-1) // 4).tolist()
    assert per_image == [112] * 4 + [88] * 8 + [48] * 4 + [50] * 2          # no re-upscaling to the 1.5 min_pixels
    assert td["image_is_lidar"].tolist() == [False] * 16 + [True] * 2
    ids = ids_of(td)
    assert count(ids, "<|image_pad|>") == 1444 and count(ids, "<|vision_start|>") == 18
    assert count(ids, "<|traj_history|>") == 48 and count(ids, "<|traj_future|>") == 128
    assert td["pixel_values"].shape[0] == int(td["image_grid_thw"].prod(-1).sum())


def test_text_layout():
    t = get_preprocess_fn(CFG)(data=sample(cot="Stop for the red light."))["text"]
    order = ["Front camera: ", "Front left camera: ", "Front right camera: ", "Front upward camera: ", "LiDAR BEV: ",
             "<|traj_history_start|>", "<|route_start|>Turn left in 11m. Speed limit 60 km/h<|route_end|>",
             "output the chain-of-thought reasoning of the driving process, then output the future trajectory.",
             "<|cot_start|>Stop for the red light.<|cot_end|>", "<|traj_future_start|>"]
    pos = [t.index(s) for s in order]
    assert pos == sorted(pos)
    assert t.count("<-0.31 seconds><|vision_start|>") == 4 and "<0.00 seconds><|vision_start|>" in t and "frame 0" not in t
    assert "<-0.21 seconds><|vision_start|>" in t and "<-0.03 seconds><|vision_start|>" in t


def test_prefix_length_generation_mode():
    td = get_preprocess_fn(CFG, generation_mode=True)(data=sample())
    ids = ids_of(td)
    assert ids[-1] == tok().convert_tokens_to_ids("<|traj_future_start|>")
    assert "output the future trajectory." in td["text"] and "<|cot_start|>" not in td["text"]
    print("prefix tokens:", len(ids))
    assert 1650 <= len(ids) <= 1850                                          # MODEL_PLAN 2.5: ~1,750
    td = get_preprocess_fn(CFG, generation_mode=True, generate_cot=True)(data=sample())
    assert td["text"].endswith("<|cot_start|>")


def test_lidar_dropped_removes_bev_block():
    td = get_preprocess_fn(CFG)(data=sample(lidar=False))
    assert "LiDAR BEV" not in td["text"] and count(ids_of(td), "<|image_pad|>") == 1344 and not td["image_is_lidar"].any()


def test_collate_label_mask():
    f = get_preprocess_fn(CFG)
    rows = []
    for s in (sample(cot="Yield to the pedestrian."), sample(lidar=False)):
        s["tokenized_data"] = f(data=s)
        for k in ("camera_frames", "bev_frames", "camera_times", "bev_times"): s.pop(k, None)
        rows.append(s)
    b = collate_fn(rows, model_config=CFG)
    ids, m = b["tokenized_data"]["input_ids"], b["labels_mask"]
    assert ids.shape[0] == 2 and b["ego_history_xyz"].shape == (2, 1, 16, 3)
    assert b["tokenized_data"]["image_is_lidar"].tolist() == [False] * 16 + [True] * 2 + [False] * 16
    T = tok()
    cot = T("Yield to the pedestrian.")["input_ids"]
    # sample 0: <cot_start> cot <cot_end> + <traj_future_start> 128 <traj_future_end> + <|im_end|>
    assert int(m[0].sum()) == len(cot) + 2 + 130 + 1
    assert int(m[1].sum()) == 130 + 1
    for t in ("<|image_pad|>", "<|traj_history|>", "<|route_start|>"):
        assert not m[ids == T.convert_tokens_to_ids(t)].any()
    assert b["tokenized_data"]["attention_mask"][1].sum() < ids.shape[1]     # left padding of the shorter sample
