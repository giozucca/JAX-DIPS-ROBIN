import os
import struct
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from pyevtk.hl import gridToVTK

casl_dir = "/Users/onur/Desktop/CASL"

def fix_vts_file(vts_path):
    print("=" * 60)
    print("Processing:", vts_path)
    
    # Read first 10000 bytes to locate binary start
    with open(vts_path, "rb") as f:
        content = f.read(10000)
        
    idx = content.find(b"<AppendedData encoding=\"raw\">\n_")
    if idx == -1:
        idx = content.find(b"<AppendedData encoding=\"raw\">_")
        if idx == -1:
            print("  Error: Could not find <AppendedData> tag")
            return
        binary_start = idx + len("<AppendedData encoding=\"raw\">_")
    else:
        binary_start = idx + len("<AppendedData encoding=\"raw\">\n_")
        
    # Find the tag end to parse XML
    app_idx = content.find(b"<AppendedData")
    tag_end = content.find(b">", app_idx)
    xml_clean = content[:tag_end + 1] + b"\n</AppendedData>\n</VTKFile>"
    
    try:
        root = ET.fromstring(xml_clean)
    except Exception as e:
        print("  Error parsing XML header:", e)
        return
        
    grid = root.find(".//StructuredGrid")
    whole_extent = grid.attrib.get("WholeExtent")
    if not whole_extent:
        print("  Error: Could not find WholeExtent")
        return
        
    ext = [int(x) for x in whole_extent.split()]
    nx = ext[1] - ext[0] + 1
    ny = ext[3] - ext[2] + 1
    nz = ext[5] - ext[4] + 1
    num_pts = nx * ny * nz
    print(f"  Grid dimensions: {nx} x {ny} x {nz} = {num_pts} points")
    
    # Parse DataArrays and their offsets
    point_data_element = root.find(".//PointData")
    if point_data_element is None:
        print("  Error: Could not find PointData element")
        return
        
    points_element = root.find(".//Points/DataArray")
    if points_element is None:
        print("  Error: Could not find Points DataArray")
        return
        
    # Build list of arrays to read
    # Each entry is (name, offset, is_points)
    arrays_to_read = [
        ("points", int(points_element.attrib["offset"]), True)
    ]
    for da in point_data_element.findall("DataArray"):
        arrays_to_read.append((da.attrib["Name"], int(da.attrib["offset"]), False))
        
    # Read the data from the file
    data_arrays = {}
    with open(vts_path, "rb") as f:
        for name, offset, is_points in arrays_to_read:
            # Seek to block start
            f.seek(binary_start + offset)
            header_bytes = f.read(8)
            block_size = struct.unpack("<Q", header_bytes)[0]
            
            # Read block
            size_to_read = num_pts * 3 * 4 if is_points else num_pts * 4
            if block_size != size_to_read:
                print(f"  Warning: block size {block_size} for {name} does not match expected {size_to_read}")
                
            raw_data = f.read(size_to_read)
            arr = np.frombuffer(raw_data, dtype=np.float32)
            
            if is_points:
                arr = arr.reshape((num_pts, 3))
                X_flat = arr[:, 0]
                Y_flat = arr[:, 1]
                Z_flat = arr[:, 2]
                
                # Coordinates coordinates are in Fortran order
                data_arrays["X"] = X_flat.reshape((nx, ny, nz), order="F").copy()
                data_arrays["Y"] = Y_flat.reshape((nx, ny, nz), order="F").copy()
                data_arrays["Z"] = Z_flat.reshape((nx, ny, nz), order="F").copy()
            else:
                # Variables are C-ordered (Z changes fastest)
                # Reshape them to 3D grid in C-order, and copy to make contiguous
                data_arrays[name] = arr.reshape((nx, ny, nz)).copy()
                
    # Create a backup
    bak_path = vts_path + ".bak"
    shutil.copy2(vts_path, bak_path)
    print(f"  Created backup at: {bak_path}")
    
    # Save the corrected data using gridToVTK
    output_prefix = os.path.splitext(vts_path)[0] # remove .vts
    
    # Extract variables for pointData
    point_data_dict = {k: v for k, v in data_arrays.items() if k not in ["X", "Y", "Z"]}
    
    gridToVTK(
        output_prefix,
        data_arrays["X"],
        data_arrays["Y"],
        data_arrays["Z"],
        pointData=point_data_dict
    )
    print(f"  Overwrote corrected VTS file at: {vts_path}")

# Find all VTS files in CASL
vts_files = []
for root, dirs, files in os.walk(casl_dir):
    for file in files:
        if file.endswith(".vts") and not file.endswith("_fixed.vts") and not file.endswith(".bak"):
            vts_files.append(os.path.join(root, file))

print(f"Found {len(vts_files)} files to fix.")
for path in sorted(vts_files):
    fix_vts_file(path)

print("Done fixing all VTS files!")
