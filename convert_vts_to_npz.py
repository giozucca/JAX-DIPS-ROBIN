import os
import glob
import struct
import numpy as np
import xml.etree.ElementTree as ET

def convert_vts_to_npz(vts_path, npz_path):
    print("Converting", vts_path)
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
        
    app_idx = content.find(b"<AppendedData")
    tag_end = content.find(b">", app_idx)
    xml_clean = content[:tag_end + 1] + b"\n</AppendedData>\n</VTKFile>"
    
    root = ET.fromstring(xml_clean)
    grid = root.find(".//StructuredGrid")
    whole_extent = grid.attrib.get("WholeExtent")
    ext = [int(x) for x in whole_extent.split()]
    nx = ext[1] - ext[0] + 1
    ny = ext[3] - ext[2] + 1
    nz = ext[5] - ext[4] + 1
    num_pts = nx * ny * nz
    
    point_data_element = root.find(".//PointData")
    points_element = root.find(".//Points/DataArray")
    
    arrays_to_read = [("points", int(points_element.attrib["offset"]), True)]
    for da in point_data_element.findall("DataArray"):
        arrays_to_read.append((da.attrib["Name"], int(da.attrib["offset"]), False))
        
    data_arrays = {}
    with open(vts_path, "rb") as f:
        for name, offset, is_points in arrays_to_read:
            f.seek(binary_start + offset)
            header_bytes = f.read(8)
            block_size = struct.unpack("<Q", header_bytes)[0]
            size_to_read = num_pts * 3 * 4 if is_points else num_pts * 4
            raw_data = f.read(size_to_read)
            arr = np.frombuffer(raw_data, dtype=np.float32)
            
            if is_points:
                arr = arr.reshape((num_pts, 3))
                data_arrays["X"] = arr[:, 0].reshape((nx, ny, nz), order="F")
                data_arrays["Y"] = arr[:, 1].reshape((nx, ny, nz), order="F")
                data_arrays["Z"] = arr[:, 2].reshape((nx, ny, nz), order="F")
            else:
                data_arrays[name] = arr.reshape((nx, ny, nz))
                
    # Save to npz
    # Rename U-U_exact to U_error
    u_error = data_arrays.get("U-U_exact", data_arrays.get("U_error", np.zeros_like(data_arrays["X"])))
    np.savez_compressed(
        npz_path, 
        X=data_arrays["X"], 
        Y=data_arrays["Y"], 
        Z=data_arrays["Z"], 
        phi=data_arrays["phi"], 
        U=data_arrays["U"], 
        U_exact=data_arrays["U_exact"], 
        U_error=u_error
    )
    print("Saved", npz_path)

if __name__ == "__main__":
    dirs = ["star_Robin3_8", "star_Robin3_16", "star_Robin3_32", "star_Robin3_64"]
    for d in dirs:
        vts_files = glob.glob(os.path.join(d, "*.vts"))
        for vts in vts_files:
            npz_path = vts.replace(".vts", ".npz")
            convert_vts_to_npz(vts, npz_path)
