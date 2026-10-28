"""Every rank must run the same graph under ZeRO-2/DDP: a training batch whose LiDAR was dropped for all samples still
gives every LiDAR-ViT parameter a (zero) gradient, and the forward result is unchanged."""
import torch

from morai_vla.model import TeacherVLA
from test_student import EXPERT, RELEASE, TINY_T, batch, need_release


@need_release
def test_all_dropped_batch_touches_lidar_vit():
    T = TeacherVLA.build("random:" + RELEASE, attn_implementation="sdpa", dtype="float32", vlm_overrides=TINY_T, expert_overrides=EXPERT,
                         train_parts=["lidar"], loss_weights=dict(traj_ce=1.0, lm=1.0, fm=0.0))
    b = batch(T, lidar=(False, False))
    keys = ("tokenized_data", "ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot", "labels_mask")
    fwd = lambda: T(**{k: (dict(v) if isinstance(v, dict) else v) for k, v in b.items() if k in keys}).loss
    T.eval(); ref = fwd()
    T.train(); loss = fwd(); loss.backward()
    lidar = list(T.vlm.model.lidar_visual.parameters())
    assert torch.allclose(loss, ref) and all(p.grad is not None and float(p.grad.abs().sum()) == 0 for p in lidar)
