"""Load Step 3 summaries; fit statistics exclusively on the caller's training mask."""
import json
from pathlib import Path

import numpy as np


def standardize(values, train, valid=None):
    """Fit on full bins only; use that same fit for both outer-half variants.

    A geometry block is fit on contact-positive bins, and absent geometry is
    exactly zero *after* normalization. Padding/missing rows never enter fits.
    """
    fit = train if valid is None else train & valid[:, :, 0]
    full = values[:, :, 0]
    if not fit.any():
        mean, std = np.zeros(full.shape[-1]), np.ones(full.shape[-1])
    else:
        mean, std = full[fit].mean(0), full[fit].std(0) + 1e-6
    out = ((values - mean) / std).astype(np.float32)
    if valid is not None:
        out *= valid[..., None]
    return out, [mean.tolist(), std.tolist()]


def enrich_store(episodes, store, heldout_videos, exclude=None, granularity=None):
    root = Path(episodes) / 'body_rich'
    if granularity is None:
        granularity = json.loads((root / 'selection.json').read_text())['granularity']
    if granularity not in (22, 32, 64):
        raise ValueError('Rich granularity must be 22, 32 or 64')
    present = store['present']
    train = present & ~np.isin(store['video_ids'], list(heldout_videos))[:, None]
    if exclude is not None:
        train &= ~exclude
    raw = {}
    for s, video in enumerate(store['video_ids']):
        with np.load(root / f'{video}.npz') as data:
            index = data['frame_idx']
            for key in (f'regions{granularity}', 'events', 'geometry', 'fingers', 'arm', 'touching'):
                value = data[key]
                raw.setdefault(key, np.zeros(present.shape + value.shape[1:], value.dtype))
                raw[key][s, index] = value
    stats = {}
    variants = {}
    contact = [raw[f'regions{granularity}']]
    events, stats['rich_events'] = standardize(raw['events'], train)
    contact.append(events)
    for hand in range(2):
        geo, stats[f'rich_geometry_{hand}'] = standardize(raw['geometry'][..., hand*11:(hand+1)*11], train,
                                                          raw['touching'][..., hand])
        contact.append(geo)
    variants['contact'] = np.concatenate(contact, axis=-1)
    for key in ('fingers', 'arm'):
        variants[key], stats[f'rich_{key}'] = standardize(raw[key], train)
    for key in ('load', 'skin'):
        if key not in store['groups']:
            raise ValueError('Rich body requires the existing physical export')
        variants[key] = np.repeat(store['groups'][key][:, :, None], 3, axis=2)
    for key in variants:
        variants[key] = variants[key].astype(np.float32)
        variants[key] *= present[..., None, None]
        if not np.isfinite(variants[key]).all():
            raise ValueError(f'Nonfinite normalized rich features: {key}')
    return {**store, 'groups': {k: v[:, :, 0] for k, v in variants.items()}, 'variants': variants,
            'representation': 'rich', 'granularity': granularity,
            'stats': {**store['stats'], **stats}, 'fit_mask': train}
