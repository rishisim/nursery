"""Run one scene-conditioned HOIDiNi interaction from an explicit GRAB prefix.

This uses the upstream feature encoder and SamplingFlow unchanged. Only its
GRAB-specific phase-one geometry lookup is scoped to the prepared scene.
No automatic prefix search, retargeting, or physics execution is performed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np

from .prepare_scene import BridgeValidationError, prepare_scene


def alignment_transform(source_object_xyz, target_object_xyz, yaw_degrees=0.0):
    """Align XY around the object while preserving the human's ground height."""
    source = np.asarray(source_object_xyz, dtype=np.float64)
    target = np.asarray(target_object_xyz, dtype=np.float64)
    if source.shape != (3,) or target.shape != (3,):
        raise BridgeValidationError("Alignment requires two XYZ positions")
    if not np.isfinite(source).all() or not np.isfinite(target).all() or not math.isfinite(yaw_degrees):
        raise BridgeValidationError("Alignment values must be finite")
    angle = math.radians(yaw_degrees)
    c, s = math.cos(angle), math.sin(angle)
    transform = np.eye(4)
    transform[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    transform[:2, 3] = (target - transform[:3, :3] @ source)[:2]
    return transform


def _output_directory(path):
    path = Path(path).expanduser().absolute()
    repo = path.parent
    while not repo.exists():
        repo = repo.parent
    ignored = subprocess.run(
        ["git", "-C", str(repo), "check-ignore", "--quiet", str(path)],
        check=False,
    )
    if ignored.returncode != 0:
        raise BridgeValidationError("Motion output must be inside an ignored repository directory")
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_motion(args):
    # Import the upstream runtime only inside its existing GPU environment.
    import torch
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from torch_geometric.data import Data
    from hoidini.amasstools.geometry import axis_angle_to_matrix, matrix_to_axis_angle
    from hoidini.amasstools.smplrifke_feats import SMPLFeatureProcessor
    from hoidini.closd.diffusion_planner.utils import dist_util
    from hoidini.closd.diffusion_planner.utils.model_util import create_gaussian_diffusion, load_saved_model
    from hoidini.closd.diffusion_planner.utils.sampler_util import ClassifierFreeSampleModel
    from hoidini.cphoi import cphoi_inference as inference
    from hoidini.cphoi.cphoi_dataset import TfmsManager, encode_cphoi_smpldata
    from hoidini.cphoi.cphoi_model import CPHOI
    from hoidini.cphoi.cphoi_utils import FeaturesDecoderWrapper
    from hoidini.datasets.dataset_smplrifke import collate_smplrifke_mdm
    from hoidini.datasets.grab.grab_utils import get_MANO_SMPLX_vertex_ids, load_mesh
    from hoidini.datasets.smpldata import SmplModelsFK
    from hoidini.datasets.smpldata_preparation import get_extended_smpldata
    from hoidini.model_utils import model_kwargs_to_device
    from hoidini.normalizer import Normalizer
    from hoidini.skeletons.mano_anchors.mano_contact_anchors import get_contact_anchors_info
    import hoidini.geometry3d.hands_intersection_loss as hand_loss

    if not torch.cuda.is_available():
        raise BridgeValidationError("Use the existing HOIDiNi CUDA environment inside a GPU allocation")
    if args.frames < 30 or args.frames > 115 or args.start_frame < 0:
        raise BridgeValidationError("This single-interaction run requires 30–115 frames and a nonnegative prefix start")
    output = _output_directory(args.output)
    (output / "cache").mkdir()
    hand_loss.CACHE_DIR_PATH = str(output / "cache")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dist_util.setup_dist(0)
    device = dist_util.dev()

    bundle = prepare_scene(args.scene, target=args.target, prompt=args.prompt, seed=args.seed)
    active = bundle.active_interaction
    mesh = active.canonical_target_mesh
    mesh_path = output / "object.obj"
    with mesh_path.open("w") as stream:
        for vertex in mesh.vertices:
            stream.write("v " + " ".join(format(v, ".10g") for v in vertex) + "\n")
        for face in mesh.faces:
            stream.write("f " + " ".join(str(int(v) + 1) for v in face) + "\n")

    cfg = OmegaConf.load(args.config)
    cfg.model_path = str(Path(args.model).resolve())
    cfg.train = OmegaConf.load(Path(args.model).resolve().parent / "args.json")
    cfg.train.model_path = cfg.model_path
    cfg.train.save_dir = str(Path(args.model).resolve().parent)
    cfg.autoregressive_include_prefix = True
    cfg.batch_size_gen = 1
    cfg.seed = args.seed
    cfg.out_dir = str(output)
    if args.steps is not None:
        if args.steps < 1:
            raise BridgeValidationError("Optimization steps must be positive")
        cfg.dno_options_phase1.num_opt_steps = args.steps
        cfg.dno_options_phase2.num_opt_steps = args.steps
    cfg.store_loss_data = False
    if cfg.train.context_len != 15 or cfg.train.fps != 20 or cfg.train.feature_names != "cphoi":
        raise BridgeValidationError("Expected the existing 15-frame, 20-fps CPHOI checkpoint")
    normalizer = Normalizer.from_dir(cfg.train.save_dir, device="cpu")
    processor = SMPLFeatureProcessor(n_pose_joints=52, n_contact_verts=60, features_string="cphoi")

    # Sixteen source frames supply the final velocity used by the 15-frame prefix.
    source = get_extended_smpldata(str(Path(args.prefix).resolve()))["smpldata"]
    prefix = source.cut(args.start_frame, args.start_frame + 16).detach()
    if len(prefix) != 16 or prefix.contact.any():
        raise BridgeValidationError("Select an explicit 16-frame pre-contact source window; grasp retargeting is out of scope")
    transform = alignment_transform(prefix.trans_obj[0], active.canonical_target_pose.translation, args.yaw)
    rotation = torch.tensor(transform[:3, :3], dtype=torch.float32)
    translation = torch.tensor(transform[:3, 3], dtype=torch.float32)
    prefix.poses[:, :3] = matrix_to_axis_angle(rotation @ axis_angle_to_matrix(prefix.poses[:, :3]))
    prefix.trans = prefix.trans @ rotation.T + translation
    prefix.joints = prefix.joints @ rotation.T + translation
    prefix.global_lhand_rotmat = rotation @ prefix.global_lhand_rotmat
    prefix.global_rhand_rotmat = rotation @ prefix.global_rhand_rotmat
    prefix.trans_obj[:] = torch.tensor(active.canonical_target_pose.translation)
    prefix.poses_obj[:] = torch.tensor(active.canonical_target_pose.axis_angle)

    # Use the scene's object surface for all 60 anchor features. Source grasp
    # labels are not transferred to a different mesh; this prefix has no contact.
    prefix_fk = SmplModelsFK.create("smplx", 16, device="cpu")
    with torch.no_grad():
        body = prefix_fk.smpldata_to_smpl_output(prefix, cancel_offset=False)
    # SMPL translation is not its pelvis position. Align the reconstructed
    # pelvis with the rotated source joints, then place the prefix's lowest
    # body vertex on the room's Z=0 floor without moving the scene object.
    pelvis_offset = prefix.joints[:, 0] - body.joints[:, 0]
    prefix.trans += pelvis_offset
    body.vertices += pelvis_offset[:, None]
    ground_shift = -float(body.vertices[:, :, 2].min())
    prefix.trans[:, 2] += ground_shift
    prefix.joints[:, :, 2] += ground_shift
    body.vertices[:, :, 2] += ground_shift
    transform[2, 3] = ground_shift
    anchor_indices = get_contact_anchors_info()[0]
    hand_indices = get_MANO_SMPLX_vertex_ids()
    indices = np.concatenate([
        np.asarray(hand_indices[hand + "_hand"])[anchor_indices] for hand in ("left", "right")
    ])
    anchors = body.vertices[:, indices]
    object_rotation = torch.tensor(active.canonical_target_pose.matrix[:3, :3], dtype=torch.float32)
    local_anchors = (anchors - prefix.trans_obj[:, None]) @ object_rotation
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    # Chunk over anchors to avoid materializing a frame × anchor × large-mesh array.
    nearest = torch.cat([torch.cdist(points[None], vertices[None]).argmin(-1) for points in local_anchors])
    prefix.local_object_points = vertices[nearest]
    features, tfms, decoded_prefix, roots = encode_cphoi_smpldata(prefix, processor)
    if not torch.isfinite(features).all():
        raise BridgeValidationError("Prefix features contain non-finite values")
    datapoint = {
        "features": normalizer.normalize(features), "text": active.prompt,
        "obj_points": Data(pos=torch.tensor(active.point_cloud.points, dtype=torch.float32),
                           normal=torch.tensor(active.point_cloud.normals, dtype=torch.float32)),
        "tfms_root_global": roots, "tfm_processor": tfms[0],
        "metadata": {"object_name": str(mesh_path), "grab_seq_path": str(args.prefix),
                     "range": (args.start_frame, args.start_frame + 15)},
    }
    _, kwargs = collate_smplrifke_mdm([datapoint], pred_len=cfg.train.pred_len, context_len=15,
                                    enforce_motion_length=15 + cfg.train.pred_len)
    model_kwargs_to_device(kwargs, device)
    assert kwargs["y"]["prefix"].shape[-1] == 15

    receipt = bundle.to_manifest()
    receipt["human_prefix"].update(provided=True, source=str(Path(args.prefix).resolve()),
                                    start_frame=args.start_frame, transform=transform.tolist(),
                                    floor_height_m=0.0,
                                    ground_shift_m=ground_shift,
                                    message="Explicit pre-contact source prefix aligned to the scene")
    receipt["ready_for_hoidini_inference"] = True
    receipt["motion_run"] = vars(args)
    (output / "input.json").write_text(json.dumps(receipt, indent=2) + "\n")
    np.savez_compressed(output / "prefix.npz", **{
        k: v.cpu().numpy() for k, v in decoded_prefix.cut(0, 15).to_dict().items() if v is not None
    })
    OmegaConf.save(cfg, output / "sampling.yaml")
    if args.prepare_only:
        print(f"Prepared prefix and scene: {output}", flush=True)
        return

    model = CPHOI(pred_len=cfg.train.pred_len, context_len=15, n_feats=normalizer.n_feats,
                  num_layers=cfg.train.layers, cond_mask_prob=cfg.train.cond_mask_prob)
    load_saved_model(model, cfg.model_path, use_avg=cfg.train.use_ema)
    if cfg.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
        kwargs["y"]["scale"] = torch.ones(1, device=device) * cfg.guidance_param
    model.freeze_object_encoder()
    model.to(device).eval()
    with torch.no_grad():
        kwargs["y"]["obj_emb"] = model.object_encoder(kwargs["y"]["obj_points"])
    normalizer = normalizer.to(device)
    decoder = FeaturesDecoderWrapper(processor, normalizer, kwargs["y"]["tfm_processor"])
    geometry = {
        "obj_v_template": [torch.tensor(active.point_cloud.points, dtype=torch.float32)],
        "table_corner_locs": [torch.tensor(active.support_corners_world, dtype=torch.float32)],
    }
    simplified_vertices, simplified_faces = load_mesh(str(mesh_path), n_simplify_faces=cfg.n_simplify_object)
    phase2_geometry = {"obj_v_template": [simplified_vertices], "obj_faces": [simplified_faces]}
    flow = inference.SamplingFlow(
        cfg, create_gaussian_diffusion(cfg.train), decoder, model,
        SmplModelsFK.create("smplx", cfg.train.pred_len + 1, device=device),
        args.seed, kwargs, [args.frames], (1, normalizer.n_feats, 1, args.frames),
        TfmsManager(processor, kwargs["y"]["tfm_processor"], normalizer),
        instantiate(cfg.dno_options_phase1), instantiate(cfg.dno_options_phase2),
        inference.get_phase1_default_dno_losses(cfg, geometry, [args.frames]),
        inference.get_phase2_default_dno_losses(cfg, torch.stack(geometry["table_corner_locs"]).to(device)),
        phase2_geometry,
    )
    # SamplingFlow otherwise reloads the GRAB table through this module-global
    # lookup. Keep this substitution scoped to the single scene run.
    with patch.object(inference, "get_geoms_batch", return_value=geometry):
        result = flow.run()
    motion = result["samples"]["final"][0].detach().to("cpu")
    arrays = {k: v.numpy() for k, v in motion.to_dict().items() if v is not None}
    np.savez_compressed(output / "motion.npz", **arrays)
    for phase, info in result["debug_data"].items():
        history = info.get("optim_info_lst")
        if history:
            inference.plot_dno_loss(history[0], save_path=str(output / f"{phase}_loss.png"))
    print(f"Saved {len(motion)} motion frames (including the 15-frame prefix): {output / 'motion.npz'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True, help="Existing upstream inference YAML")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--start-frame", type=int, default=0, help="Index after upstream resampling to 20 fps")
    parser.add_argument("--yaw", type=float, default=0.0, help="Explicit human heading adjustment, in degrees")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--steps", type=int, help="Optional override; otherwise retain upstream optimization settings")
    parser.add_argument("--seed", type=int, default=90323)
    parser.add_argument("--prepare-only", action="store_true")
    run_motion(parser.parse_args())


if __name__ == "__main__":
    main()
