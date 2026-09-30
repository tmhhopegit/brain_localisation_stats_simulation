"""Reading and writing brain images (copied from the ploras package, so this one stands alone).

Replaces spm_vol / spm_read_vols, rdir.m, niftiread, imresize3 and the file searches
in GetBrains.m, get_ims_smoothed.m and PLORAS.m's find_gm_name.

Uses nibabel if it is installed. Otherwise a small built-in reader handles NIfTI-1 (.nii,
.nii.gz) and NIfTI / Analyze 7.5 pairs (.img + .hdr), which covers SPM's outputs. Arrays
are indexed [x, y, z] like spm_read_vols.
"""
from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

_DTYPES = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64, 256: np.int8,
           512: np.uint16, 768: np.uint32, 1024: np.int64, 1280: np.uint64}


def _open(path):
    path = str(path)
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def _read_header(raw: bytes):
    endian = "<" if struct.unpack("<i", raw[:4])[0] == 348 else ">"
    if struct.unpack(endian + "i", raw[:4])[0] != 348:
        raise ValueError("not a NIfTI-1 / Analyze header")
    dim = struct.unpack(endian + "8h", raw[40:56])
    datatype, bitpix = struct.unpack(endian + "2h", raw[70:74])
    pixdim = struct.unpack(endian + "8f", raw[76:108])
    vox_offset = struct.unpack(endian + "f", raw[108:112])[0]
    slope, inter = struct.unpack(endian + "2f", raw[112:120])
    qform_code, sform_code = struct.unpack(endian + "2h", raw[252:256])
    quat = struct.unpack(endian + "6f", raw[256:280])
    srow = np.array(struct.unpack(endian + "12f", raw[280:328])).reshape(3, 4)
    magic = raw[344:348]
    shape = tuple(int(d) for d in dim[1:1 + max(dim[0], 3)])
    if sform_code > 0:
        affine = np.vstack([srow, [0, 0, 0, 1]])
    elif qform_code > 0:
        affine = _qform_affine(quat, pixdim)
    else:
        # Analyze 7.5 without orientation: SPM's default assumption (voxel sizes, origin at centre,
        # x flipped as in SPM's "neurological -> radiological" default for Analyze)
        vs = np.abs(np.array(pixdim[1:4]))
        vs[vs == 0] = 1
        origin = (np.array(shape[:3]) + 1) / 2
        affine = np.diag([-vs[0], vs[1], vs[2], 1.0])
        affine[:3, 3] = -affine[:3, :3] @ (origin - 1)
    return dict(endian=endian, shape=shape, datatype=datatype, vox_offset=vox_offset, slope=slope, inter=inter,
                affine=affine, pair=magic[:3] in (b"ni1", b"\x00\x00\x00") or magic == b"\x00\x00\x00\x00")


def _qform_affine(quat, pixdim):
    b, c, d, qx, qy, qz = quat
    a = np.sqrt(max(0.0, 1 - (b * b + c * c + d * d)))
    R = np.array([[a*a + b*b - c*c - d*d, 2*b*c - 2*a*d, 2*b*d + 2*a*c],
                  [2*b*c + 2*a*d, a*a + c*c - b*b - d*d, 2*c*d - 2*a*b],
                  [2*b*d - 2*a*c, 2*c*d + 2*a*b, a*a + d*d - c*c - b*b]])
    qfac = -1.0 if pixdim[0] < 0 else 1.0
    Z = np.diag([pixdim[1], pixdim[2], pixdim[3] * qfac])
    A = np.eye(4)
    A[:3, :3] = R @ Z
    A[:3, 3] = [qx, qy, qz]
    return A


def read_image(path) -> tuple[np.ndarray, np.ndarray]:
    """(data as float64 [x, y, z(, t)], 4x4 voxel-to-world affine)."""
    path = Path(path)
    try:
        import nibabel as nib
        img = nib.load(str(path))
        return np.asarray(img.get_fdata(), dtype=np.float64), img.affine
    except ImportError:
        pass
    name = path.name.lower()
    if name.endswith((".img", ".img.gz", ".hdr", ".hdr.gz")):
        stem = str(path)[: -len(".img.gz")] if name.endswith((".img.gz", ".hdr.gz")) else str(path)[:-4]
        gz = ".gz" if name.endswith(".gz") else ""
        hdr_path, img_path = Path(stem + ".hdr" + gz), Path(stem + ".img" + gz)
        with _open(hdr_path) as f:
            h = _read_header(f.read(348))
        with _open(img_path) as f:
            raw = f.read()
        offset = 0
    else:
        with _open(path) as f:
            raw = f.read()
        h = _read_header(raw[:348])
        offset = int(h["vox_offset"]) or 352
    dtype = np.dtype(_DTYPES[h["datatype"]]).newbyteorder(h["endian"])
    n = int(np.prod(h["shape"]))
    data = np.frombuffer(raw, dtype=dtype, count=n, offset=offset).reshape(h["shape"], order="F").astype(np.float64)
    if h["slope"] not in (0.0,) and np.isfinite(h["slope"]) and (h["slope"] != 1 or h["inter"] != 0):
        data = data * h["slope"] + h["inter"]
    return data, h["affine"]


def write_image(path, data, affine=None) -> None:
    """Write a float32 NIfTI-1 (.nii or .nii.gz); used for synthetic data and outputs."""
    data = np.asarray(data, dtype=np.float32)
    affine = np.eye(4) if affine is None else np.asarray(affine, dtype=np.float64)
    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)
    dims = [data.ndim] + list(data.shape) + [1] * (7 - data.ndim)
    struct.pack_into("<8h", hdr, 40, *dims)
    struct.pack_into("<2h", hdr, 70, 16, 32)
    vs = np.sqrt((affine[:3, :3] ** 2).sum(0))
    struct.pack_into("<8f", hdr, 76, 1.0, *vs, 1, 1, 1, 1)
    struct.pack_into("<f", hdr, 108, 352.0)
    struct.pack_into("<2f", hdr, 112, 1.0, 0.0)
    struct.pack_into("<2h", hdr, 252, 0, 2)                       # qform 0, sform 2 (aligned)
    struct.pack_into("<12f", hdr, 280, *affine[:3].ravel())
    hdr[344:348] = b"n+1\x00"
    payload = bytes(hdr) + b"\x00" * 4 + data.tobytes(order="F")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(str(path), "wb") as f:
        f.write(payload)


def find_image(folder, patterns, recursive: bool = False, pick: str = "first") -> Path | None:
    """First (or last) file in `folder` matching any of `patterns`, tried in order.

    E.g. find_image(lesion_folder, ["swc1*.img", "swc1*.nii"]) is PLORAS.m's find_gm_name;
    GetBrains.m used recursive=True, pick='last'. Returns None if nothing matches (the
    original raised an error or indexed an empty list)."""
    folder = Path(folder)
    for pat in ([patterns] if isinstance(patterns, str) else patterns):
        hits = sorted(folder.rglob(pat) if recursive else folder.glob(pat))
        hits = [h for h in hits if h.suffix.lower() in (".nii", ".img", ".gz")]
        if hits:
            return hits[0] if pick == "first" else hits[-1]
    return None


def resize(volume, shape) -> np.ndarray:
    """Resample a 3D volume to `shape` with linear interpolation (imresize3)."""
    from scipy.ndimage import zoom
    volume = np.asarray(volume, dtype=np.float64)
    factors = [s / v for s, v in zip(shape, volume.shape)]
    out = zoom(volume, factors, order=1)
    return out[tuple(slice(0, s) for s in shape)]


def world_x(affine, shape) -> np.ndarray:
    """World x coordinate (mm) of every voxel, flattened in Fortran order like the images."""
    i = np.arange(shape[0])
    x = affine[0, 0] * i + affine[0, 3]
    if affine[0, 1] != 0 or affine[0, 2] != 0:
        ii, jj, kk = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
        return (affine[0, 0] * ii + affine[0, 1] * jj + affine[0, 2] * kk + affine[0, 3]).ravel(order="F")
    return np.broadcast_to(x[:, None, None], shape[:3]).ravel(order="F")
