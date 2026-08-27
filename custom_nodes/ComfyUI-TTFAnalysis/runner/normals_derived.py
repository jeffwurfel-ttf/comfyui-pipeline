"""
Surface normals from depth, on GPU. Mirrors the P1 numpy reference op-for-op.

The CPU version was the slowest signal in P1 at 111 s for a 398-frame shot,
against 18 s for depth inference itself — absurd for what is a cross product of
two finite differences. Same math, torch, float32.

Kept deliberately parallel to the numpy reference so the two can be diffed
directly; runner.tests.test_normals_parity asserts they agree.
"""
import numpy as np
import torch


# Orientation tiebreak band. P1 oriented normals with a bare `n_z > 0` test.
# For a surface exactly edge-on to the camera n_z is ~0 and the test sits on a
# knife edge: float-ordering differences of ~1e-8 between two implementations
# (or two numpy builds) flip such a normal by a full 180 degrees. Measured on
# the mountaindemo shot that is 0.0184% of pixels, all with |n_z| <= 5.3e-5.
# Inside the band the orientation is genuinely ambiguous — both +n and -n are
# valid — so it is settled on n_y instead, which is O(1) there and therefore
# stable. Outside the band behaviour is identical to P1.
# Inside the band the sign is taken from whichever of n_y / n_x is larger in
# magnitude. Since |n| == 1 and |n_z| is tiny there, n_x^2 + n_y^2 ~= 1, so the
# chosen component is always >= 0.707 and its sign is numerically solid. A first
# attempt keyed the band on n_y alone and still flipped 339 px, because n_y has
# its own zero crossing for surfaces normal to the x axis.
ORIENT_EPS = 1e-4


def _flip_mask_t(n, eps):
    nx, ny, nz = n[..., 0:1], n[..., 1:2], n[..., 2:3]
    degenerate = torch.where(ny.abs() >= nx.abs(), ny > 0, nx > 0)
    return torch.where(nz.abs() > eps, nz > 0, degenerate)


def _flip_mask_np(n, eps):
    nx, ny, nz = n[..., 0:1], n[..., 1:2], n[..., 2:3]
    degenerate = np.where(np.abs(ny) >= np.abs(nx), ny > 0, nx > 0)
    return np.where(np.abs(nz) > eps, nz > 0, degenerate)


def _minabs_diff_np(P, axis):
    f = np.diff(P, axis=axis)
    pad = [(0, 0)] * P.ndim
    pad_f = list(pad); pad_f[axis] = (0, 1)
    pad_b = list(pad); pad_b[axis] = (1, 0)
    fwd = np.pad(f, pad_f, mode="edge")
    bwd = np.pad(f, pad_b, mode="edge")
    return np.where(np.abs(fwd[..., 2:3]) <= np.abs(bwd[..., 2:3]), fwd, bwd)


def normals_from_depth_cpu(disp, lo, hi, fov_deg=60.0, edge_rel=0.05):
    """numpy reference, kept in this module so the two paths cannot drift.
    Identical math to normals_from_depth_gpu, including the orientation band."""
    disp = np.asarray(disp, np.float32)
    t, H, W = disp.shape
    f = 0.5 * W / np.tan(0.5 * np.deg2rad(fov_deg))
    u = (np.arange(W, dtype=np.float32) - W * 0.5)[None, None, :]
    v = (np.arange(H, dtype=np.float32) - H * 0.5)[None, :, None]
    dn = np.clip((disp - lo) / max(hi - lo, 1e-6), 1e-3, 1.0)
    z = (1.0 / dn).astype(np.float32)
    P = np.stack([u * z / f, v * z / f, z], -1)
    n = np.cross(_minabs_diff_np(P, 2), _minabs_diff_np(P, 1))
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(ln, 1e-8)
    n = np.where(_flip_mask_np(n, ORIENT_EPS), -n, n)
    gx = np.abs(np.diff(z, axis=2, append=z[:, :, -1:])) / np.maximum(z, 1e-6)
    gy = np.abs(np.diff(z, axis=1, append=z[:, -1:, :])) / np.maximum(z, 1e-6)
    M = ((np.maximum(gx, gy) < edge_rel) & (ln[..., 0] > 1e-8)).astype(np.uint8)
    return n.transpose(0, 3, 1, 2).astype(np.float32), M


def _minabs_diff_t(P, axis):
    """Forward/backward difference, elementwise whichever has the smaller depth
    step — never differentiates across a silhouette. Matches np.pad(mode='edge')
    on the numpy side: the forward field replicates its last slice, the backward
    field replicates its first."""
    f = torch.diff(P, dim=axis)
    if axis == 2:
        fwd = torch.cat([f, f[:, :, -1:, :]], dim=2)
        bwd = torch.cat([f[:, :, :1, :], f], dim=2)
    elif axis == 1:
        fwd = torch.cat([f, f[:, -1:, :, :]], dim=1)
        bwd = torch.cat([f[:, :1, :, :], f], dim=1)
    else:
        raise ValueError(axis)
    take_f = fwd[..., 2:3].abs() <= bwd[..., 2:3].abs()
    return torch.where(take_f, fwd, bwd)


def normals_from_depth_gpu(disp, lo, hi, fov_deg=60.0, edge_rel=0.05,
                           device="cuda", out_np=True):
    """disp: (t,H,W) float32 inverse depth (larger = nearer), one chunk.

    `lo`/`hi` are the SHOT-scoped percentiles, passed in rather than computed
    here — that is what makes this streamable. Computing them per chunk would
    silently reintroduce per-chunk normalization, which is the same class of bug
    as per-frame depth normalization in P2.

    Returns (t,3,H,W) unit normals and (t,H,W) uint8 validity mask.
    """
    if not torch.is_tensor(disp):
        disp = torch.from_numpy(np.ascontiguousarray(disp))
    d = disp.to(device=device, dtype=torch.float32)
    t, H, W = d.shape

    f = 0.5 * W / np.tan(0.5 * np.deg2rad(fov_deg))
    u = (torch.arange(W, device=device, dtype=torch.float32) - W * 0.5).view(1, 1, W)
    v = (torch.arange(H, device=device, dtype=torch.float32) - H * 0.5).view(1, H, 1)

    dn = ((d - lo) / max(hi - lo, 1e-6)).clamp(1e-3, 1.0)
    z = 1.0 / dn
    P = torch.stack([u * z / f, v * z / f, z], dim=-1)          # (t,H,W,3)

    n = torch.linalg.cross(_minabs_diff_t(P, 2), _minabs_diff_t(P, 1), dim=-1)
    ln = torch.linalg.norm(n, dim=-1, keepdim=True)
    n = n / ln.clamp_min(1e-8)
    n = torch.where(_flip_mask_t(n, ORIENT_EPS), -n, n)         # face the camera

    zx = torch.cat([z, z[:, :, -1:]], dim=2)
    zy = torch.cat([z, z[:, -1:, :]], dim=1)
    gx = (zx[:, :, 1:] - zx[:, :, :-1]).abs() / z.clamp_min(1e-6)
    gy = (zy[:, 1:, :] - zy[:, :-1, :]).abs() / z.clamp_min(1e-6)
    M = ((torch.maximum(gx, gy) < edge_rel) & (ln[..., 0] > 1e-8)).to(torch.uint8)

    N = n.permute(0, 3, 1, 2).contiguous()
    if out_np:
        return N.cpu().numpy(), M.cpu().numpy()
    return N, M
