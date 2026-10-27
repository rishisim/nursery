"""Body windows, missing-data masks and minibatch context shared by learners."""

import numpy as np
import torch

from .losses import REGIONS


class BodyData:
    """The body store, pair metadata and episode bags on one device; builds the per-batch context.

    pairs: ps/pt session row and centre frame per training pair [N]; win_frames [N,Fmax] frames of the pair's
    image window (-1 pad, same order as upstream FramePairs); bags [S,T,E] frames of each second's episode (-1 none),
    bag_episode [S,T] its episode id;
    cache/row_of: the region cache in RAM and its row per (session, frame), to fetch bag images.
    """

    def __init__(self, store, ps, pt, win_frames, bags, bag_episode, cache, row_of, device, half=5, mask_prob=0.0, dropout=0.5):
        self.store, self.half, self.mask_prob, self.dropout, self.device = store, half, mask_prob, dropout, device
        t = lambda a, dtype=None: torch.as_tensor(a, dtype=dtype, device=device)
        self.groups = {k: t(v) for k, v in store["groups"].items()}
        self.targets = {k: t(v) for k, v in store["targets"].items()}
        self.clip, self.present, self.next_valid = t(store["clip"]), t(store["present"]), t(store["next_valid"])
        self.ps, self.pt, self.win_frames = t(ps), t(pt), t(win_frames)
        self.bags, self.bag_episode, self.cache, self.row_of = bags, bag_episode, cache, row_of
        self.offsets = torch.arange(-half, half + 1, device=device)
        self.dims = {k: v.shape[-1] for k, v in store["groups"].items()}

    def window(self, s, t):
        """Window inputs for sessions s and centre frames t (device long tensors)."""
        steps = t[:, None] + self.offsets
        inside = (steps >= 0) & (steps < self.clip.shape[1])
        steps = steps.clamp(0, self.clip.shape[1] - 1)
        seg = torch.where(inside & self.present[s[:, None], steps], self.clip[s[:, None], steps],
                          torch.full_like(steps, -2))
        groups = {k: g[s[:, None], steps] for k, g in self.groups.items()}
        return groups, (seg == -2).float(), seg

    def batch(self, idx, generator, training=True, bag_frames=9):
        """Context for pair indices `idx` (numpy/long): window inputs, vision targets, bags, masked-modelling mask."""
        idx = torch.as_tensor(idx, device=self.device)
        s, t = self.ps[idx], self.pt[idx]
        groups, missing, seg = self.window(s, t)
        half = self.half
        same = (seg == seg[:, half:half + 1]) & (seg >= 0)
        ctx = {"groups": groups, "missing": missing, "seg": seg, "same": same}
        # body dropout: a dropped pair is scored from the picture alone
        ctx["keep"] = ((torch.rand(len(s), generator=generator) >= self.dropout).float().to(self.device)
                       if training else torch.ones(len(s), device=self.device))
        if training and self.mask_prob > 0:
            pick = torch.rand(seg.shape, generator=generator).to(self.device) < self.mask_prob
            ctx["mask"] = pick & same
        wf = self.win_frames[idx]
        valid = wf >= 0
        fr = wf.clamp(min=0)
        ctx["frame_valid"] = valid
        ctx["frame_targets"] = {k: v[s[:, None], fr] for k, v in self.targets.items()}
        ctx["frame_next_valid"] = self.next_valid[s[:, None], fr] & valid
        pick = []
        if self.bags is not None:
            bag = torch.as_tensor(self.bags[s.cpu().numpy(), t.cpu().numpy()])  # [B,E] frames
            pick = torch.nonzero((bag >= 0).sum(1) >= 2).squeeze(1)
        if len(pick):
            frames = bag[pick].numpy()
            rows = self.row_of[s.cpu().numpy()[pick.numpy()][:, None], np.maximum(frames, 0)]
            images = torch.from_numpy(self.cache[rows.reshape(-1)].astype(np.float32)).to(self.device)
            ctx["bag_sel"] = pick.to(self.device)
            ctx["bag_images"] = images.view(len(pick), frames.shape[1], REGIONS, -1)
            ctx["bag_valid"] = torch.from_numpy(frames >= 0).to(self.device)
            ctx["bag_episode"] = torch.as_tensor(self.bag_episode[s.cpu().numpy()[pick.numpy()], t.cpu().numpy()[pick.numpy()]]).to(self.device)
            # two distinct frames of each bag for the consistency loss
            n = ctx["bag_valid"].sum(1)
            a = (torch.rand(len(pick), generator=generator) * n.cpu()).long()
            b = (a + 1 + (torch.rand(len(pick), generator=generator) * (n.cpu() - 1)).long()) % n.cpu()
            ctx["bag_ab"] = torch.stack([a, b], 1).to(self.device)
        return ctx
