"""
Shrink a JAX-DIPS .vts output down to the region you actually look at.

The solver writes the full evaluation grid (Nx_eval^3, e.g. 128^3 = 2.1M points,
~56 MB with four fields). For a Robin problem only phi <= 0 is meaningful -- the
Omega+ values come from an untrained network -- and that is typically 5-15% of the
box. This script:

  1. crops the structured grid to the bounding box of {phi <= 0} (+ a margin),
  2. optionally sets every field to NaN where phi > 0, so ParaView hides the
     exterior automatically and the colour range is set by the interior alone,
  3. optionally strides the grid to downsample further.

Usage
    python3 tools/crop_vts.py in.vts out           # writes out.vts
    python3 tools/crop_vts.py in.vts out --stride 2
    python3 tools/crop_vts.py in.vts out --no-blank --margin 4

Needs only numpy + pyevtk (both already installed).
"""
import argparse
import os
import re
import sys

import numpy as np
from pyevtk.hl import gridToVTK

NP_DTYPE = {"Float32": np.float32, "Float64": np.float64,
            "Int32": np.int32, "Int64": np.int64, "UInt64": np.uint64}


def read_vts(path):
    """Minimal reader for pyevtk-written .vts (appended raw, UInt64 headers)."""
    with open(path, "rb") as f:
        raw = f.read()
    marker = raw.find(b"<AppendedData")
    if marker < 0:
        raise ValueError("no <AppendedData> section -- not a pyevtk .vts?")
    header = raw[:marker].decode("utf-8", "replace")
    start = raw.find(b"_", marker) + 1          # data begins right after '_'

    ext = re.search(r'WholeExtent="([\d\s]+)"', header).group(1).split()
    x0, x1, y0, y1, z0, z1 = (int(v) for v in ext)
    dims = (x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1)

    hdr_t = NP_DTYPE[re.search(r'header_type="(\w+)"', header).group(1)]
    hdr_sz = np.dtype(hdr_t).itemsize

    arrays, points = {}, None
    for m in re.finditer(
        r'<DataArray Name="([^"]+)" NumberOfComponents="(\d+)" type="(\w+)"'
        r'[^>]*offset="(\d+)"', header
    ):
        name, ncomp, dtype, off = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        base = start + off
        nbytes = int(np.frombuffer(raw[base:base + hdr_sz], dtype=hdr_t, count=1)[0])
        buf = np.frombuffer(raw[base + hdr_sz:base + hdr_sz + nbytes], dtype=NP_DTYPE[dtype])
        if name == "points":
            points = buf.reshape(-1, 3)
        else:
            # pyevtk writes point data in Fortran order
            arrays[name] = buf.reshape(dims, order="F")
    if points is None:
        raise ValueError("no 'points' array found")
    return dims, points, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outstem", help="output path WITHOUT the .vts extension")
    ap.add_argument("--field", default="phi", help="level-set field name (default: phi)")
    ap.add_argument("--margin", type=int, default=2, help="extra cells around the crop box")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth point per axis")
    ap.add_argument("--no-blank", action="store_true",
                    help="keep exterior values instead of setting them to NaN")
    a = ap.parse_args()

    dims, points, arrays = read_vts(a.infile)
    n_in = dims[0] * dims[1] * dims[2]
    print(f"read {a.infile}: {dims[0]}x{dims[1]}x{dims[2]} = {n_in:,} points, "
          f"fields {sorted(arrays)}  ({os.path.getsize(a.infile)/1e6:.1f} MB)")

    if a.field not in arrays:
        sys.exit(f"field '{a.field}' not in file; available: {sorted(arrays)}")
    phi = arrays[a.field]

    inside = phi <= 0.0
    if not inside.any():
        sys.exit(f"no point has {a.field} <= 0 -- nothing to crop to")
    idx = [np.where(inside.any(axis=tuple(j for j in range(3) if j != i)))[0]
           for i in range(3)]
    lo = [max(0, int(i[0]) - a.margin) for i in idx]
    hi = [min(dims[k] - 1, int(idx[k][-1]) + a.margin) for k in range(3)]
    sl = tuple(slice(lo[k], hi[k] + 1, a.stride) for k in range(3))
    print(f"  {a.field} <= 0 at {int(inside.sum()):,} points "
          f"({100*inside.sum()/n_in:.1f}%); cropping to "
          f"[{lo[0]}:{hi[0]}, {lo[1]}:{hi[1]}, {lo[2]}:{hi[2]}] stride {a.stride}")

    X = points[:, 0].reshape(dims, order="F")[sl]
    Y = points[:, 1].reshape(dims, order="F")[sl]
    Z = points[:, 2].reshape(dims, order="F")[sl]

    out = {}
    keep = arrays[a.field][sl] <= 0.0
    for name, val in arrays.items():
        v = np.ascontiguousarray(val[sl], dtype=np.float32)
        if not a.no_blank and name != a.field:
            v = np.where(keep, v, np.nan).astype(np.float32)
        out[name] = v

    gridToVTK(a.outstem,
              np.ascontiguousarray(X, dtype=np.float32),
              np.ascontiguousarray(Y, dtype=np.float32),
              np.ascontiguousarray(Z, dtype=np.float32),
              pointData=out)

    n_out = out[a.field].size
    path = a.outstem + ".vts"
    print(f"wrote {path}: {out[a.field].shape} = {n_out:,} points "
          f"({os.path.getsize(path)/1e6:.1f} MB)  --  {n_in/n_out:.1f}x fewer points")
    if not a.no_blank:
        print("  exterior set to NaN: ParaView will hide it and autoscale to the interior")


if __name__ == "__main__":
    main()
