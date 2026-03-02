
import os
import argparse
import csv
import sys
import numpy as np
import tensorflow as tf
import h5py
from sionna.rt import load_scene, SceneObject, Transmitter, Receiver, PlanarArray, RadioMaterial, PathSolver
from sionna.rt.scene import box as box_xml_path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.sim.config import load_config

def generate_factory_dataset(args):
    """
    Generates a synthetic dataset for a 6G smart factory environment using Sionna.
    """
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
    config = load_config(config_path).to_dict()
    
    factory_params = config.get("factory_scenario", {})
    system_params = config.get("system", {})
    rt_params = config.get("ray_tracing", {})
    tr_params = config.get("transceiver", {})
    
    # Check GPU availability
    gpu_num = config.get("simulation", {}).get("gpu_id", 0)
    if args.gpu >= 0:
        gpu_num = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

    # 1. Generate XML Scene File
    # Robustly find box.ply
    try:
        box_dir = os.path.dirname(box_xml_path)
        box_ply_path = os.path.join(box_dir, "meshes", "box.ply")
    except Exception as e:
        print(f"Error locating box scene path: {e}")
        return

    if not os.path.exists(box_ply_path):
        print(f"Error: box.ply not found at {box_ply_path}. Cannot generate factory geometry.")
        return

    xml_content = ['<scene version="2.1.0">']
    xml_content.append('    <bsdf type="itu-radio-material" id="mat_default"> <string name="type" value="concrete"/> </bsdf>')
    
    dims = factory_params.get("room_dimensions", [20.0, 20.0, 6.0])
    L, W, H = dims[0], dims[1], dims[2]
    
    def add_shape_xml(name, size, pos):
        # Normalize scale based on 10x10x5 base mesh
        sx = size[0] / 10.0
        sy = size[1] / 10.0
        sz = size[2] / 5.0
        xml_content.append(f'    <shape type="ply" id="{name}">')
        xml_content.append(f'        <string name="filename" value="{box_ply_path}"/>')
        xml_content.append(f'        <ref id="mat_default" name="bsdf"/>')
        xml_content.append('        <transform name="to_world">')
        xml_content.append(f'            <scale x="{sx}" y="{sy}" z="{sz}"/>')
        xml_content.append(f'            <translate x="{pos[0]}" y="{pos[1]}" z="{pos[2]}"/>')
        xml_content.append('        </transform>')
        xml_content.append('    </shape>')

    thickness = tr_params.get("wall_thickness", 0.2)
    # Floor sits below z=0
    add_shape_xml("floor", [L, W, thickness], [0, 0, -thickness])
    # Ceiling sits above z=H
    add_shape_xml("ceiling", [L, W, thickness], [0, 0, H])
    
    # Walls
    add_shape_xml("wall_left", [thickness, W, H], [-L/2 - thickness/2, 0, 0])
    add_shape_xml("wall_right", [thickness, W, H], [L/2 + thickness/2, 0, 0])
    add_shape_xml("wall_front", [L, thickness, H], [0, -W/2 - thickness/2, 0])
    add_shape_xml("wall_back", [L, thickness, H], [0, W/2 + thickness/2, 0])
    
    # Machines
    num_machines = factory_params.get("num_machines", 5)
    machine_ranges = factory_params.get("machine_size_range", [[1.0, 3.0], [1.0, 3.0], [1.0, 2.5]])
    room_padding = tr_params.get("room_padding", 2.0)
    
    print(f"Generating factory: {L}x{W}x{H}m with {num_machines} machines.")
    for i in range(num_machines):
        sx_target = np.random.uniform(machine_ranges[0][0], machine_ranges[0][1])
        sy_target = np.random.uniform(machine_ranges[1][0], machine_ranges[1][1])
        sz_target = np.random.uniform(machine_ranges[2][0], machine_ranges[2][1])
        
        px = np.random.uniform(-L/2 + room_padding, L/2 - room_padding)
        py = np.random.uniform(-W/2 + room_padding, W/2 - room_padding)
        add_shape_xml(f"machine_{i}", [sx_target, sy_target, sz_target], [px, py, 0])
        
    xml_content.append('</scene>')
    
    scene_filename = f"factory_scene_{os.getpid()}.xml"
    with open(scene_filename, 'w') as f:
        f.write('\n'.join(xml_content))
        
    # 2. Load the Generated Scene
    scene = load_scene(scene_filename)
    
    # 3. Apply Materials
    materials_config = factory_params.get("materials", {})
    metal_cfg = materials_config.get("metal", {"name": "factory_metal", "relative_permittivity": 1.0, "conductivity": 1e7})
    metal = RadioMaterial(metal_cfg["name"], relative_permittivity=metal_cfg["relative_permittivity"], conductivity=metal_cfg["conductivity"])

    concrete_cfg = materials_config.get("concrete", {"name": "factory_concrete", "relative_permittivity": 7.0, "conductivity": 0.1})
    concrete = RadioMaterial(concrete_cfg["name"], relative_permittivity=concrete_cfg["relative_permittivity"], conductivity=concrete_cfg["conductivity"])

    if metal.name not in scene.radio_materials: scene.add(metal)
    if concrete.name not in scene.radio_materials: scene.add(concrete)
        
    for name, obj in scene.objects.items():
        if name.startswith("machine"):
            obj.radio_material = metal.name
        else:
            obj.radio_material = concrete.name

    if "mat_default" in scene.radio_materials:
        try:
            scene.remove("mat_default")
        except:
            pass

    # 4. Set Frequency & Compute
    scene.frequency = system_params.get("carrier_frequency", 140e9)
    # Frequency can also be overridden by command line arg if we wanted, but let's stick to config
    scene.synthetic_array = True
    
    # Configure TX/RX
    tx_pos = [0, 0, H - tr_params.get("tx_height_offset", 1.0)]
    num_bs_ant = system_params.get("num_bs_ant", 64)
    rows = int(np.sqrt(num_bs_ant))
    cols = num_bs_ant // rows
    ant_spacing = tr_params.get("antenna_spacing", 0.5)
    
    scene.tx_array = PlanarArray(num_rows=rows, num_cols=cols, 
                                 vertical_spacing=ant_spacing, horizontal_spacing=ant_spacing, 
                                 pattern=tr_params.get("tx_pattern", "tr38901"), 
                                 polarization=tr_params.get("tx_polarization", "VH"))
    tx = Transmitter("BS", position=tx_pos)
    scene.add(tx)

    num_ut_ant = system_params.get("num_ut_ant", 1)
    rx_rows = int(np.sqrt(num_ut_ant))
    rx_cols = num_ut_ant // rx_rows 
    if rx_cols == 0: rx_cols = 1
    scene.rx_array = PlanarArray(num_rows=rx_rows, num_cols=rx_cols, 
                                 vertical_spacing=ant_spacing, horizontal_spacing=ant_spacing, 
                                 pattern=tr_params.get("rx_pattern", "iso"), 
                                 polarization=tr_params.get("rx_polarization", "V"))
    rx = Receiver("RX", position=[0,0,tr_params.get("rx_height", 1.0)])
    scene.add(rx)

    # Computation Loop
    num_samples = args.num_samples
    all_cir = []
    all_pos = []
    solver = PathSolver()
    
    max_depth = rt_params.get("max_depth", args.max_depth)
    samples_per_src = rt_params.get("samples_per_src", 1000000)
    MAX_PATHS = rt_params.get("max_paths", 100)
    rx_padding = tr_params.get("rx_boundary_padding", 1.0)
    
    print(f"Starting dataset generation for {num_samples} samples...")
    for i in range(num_samples):
        # Move RX
        rx_x = np.random.uniform(-L/2 + rx_padding, L/2 - rx_padding)
        rx_y = np.random.uniform(-W/2 + rx_padding, W/2 - rx_padding)
        rx.position = [rx_x, rx_y, tr_params.get("rx_height", 1.0)]
        
        # Ray Tracing
        paths = solver(scene, max_depth=max_depth, samples_per_src=int(samples_per_src)) 
        a_val, tau_val = paths.cir(out_type="numpy")
        
        # Standardize number of paths
        # a_val: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, time]
        # tau_val: [num_rx, num_tx, num_paths]
        p_axis_a = 4
        p_axis_tau = 2
        
        curr_paths = a_val.shape[p_axis_a]
        if curr_paths < MAX_PATHS:
            pad_shape_a = list(a_val.shape); pad_shape_a[p_axis_a] = MAX_PATHS - curr_paths
            a_val = np.concatenate([a_val, np.zeros(pad_shape_a, dtype=a_val.dtype)], axis=p_axis_a)
            pad_shape_tau = list(tau_val.shape); pad_shape_tau[p_axis_tau] = MAX_PATHS - curr_paths
            tau_val = np.concatenate([tau_val, -np.ones(pad_shape_tau, dtype=tau_val.dtype)], axis=p_axis_tau)
        else:
            a_val = a_val[:,:,:,:,:MAX_PATHS,:]
            tau_val = tau_val[:,:,:MAX_PATHS]
        
        all_cir.append( (a_val, tau_val) ) 
        all_pos.append( [rx_x, rx_y, tr_params.get("rx_height", 1.0)] )
        if i % 10 == 0: print(f"Generated sample {i}/{num_samples}")

    # Save
    output_path = os.path.join("data", args.output_filename)
    os.makedirs("data", exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["frequency"] = scene.frequency
        f.create_dataset("rx_positions", data=np.array(all_pos))
        if len(all_cir) > 0:
             a_list = [c[0] for c in all_cir]
             tau_list = [c[1] for c in all_cir]
             f.create_dataset("paths_a", data=np.concatenate(a_list, axis=0))
             f.create_dataset("paths_tau", data=np.concatenate(tau_list, axis=0))
    print(f"Dataset saved to {output_path}")

    # 5. Save to CSV (Flattened/Summary version)
    csv_output_path = output_path.replace(".h5", ".csv")
    print(f"Exporting summary data to CSV: {csv_output_path}")
    
    try:
        with open(csv_output_path, 'w', newline='') as csvfile:
            # Defined columns: Sample ID, Rx Pos (x,y,z), Top 10 Paths (Mag, Delay)
            fieldnames = ['sample_id', 'pos_x', 'pos_y', 'pos_z']
            NUM_CSV_PATHS = 10
            for p in range(NUM_CSV_PATHS):
                fieldnames.append(f'path_{p}_mag')
                fieldnames.append(f'path_{p}_delay')
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i in range(len(all_cir)):
                a_val, tau_val = all_cir[i]
                pos = all_pos[i]
                
                # a_val shape: [1, num_rx_ant, 1, num_tx_ant, paths, 1]
                # Compute magnitude (average over antennas)
                # Squeeze to [num_rx_ant, num_tx_ant, paths]
                # Then mean over antennas
                try:
                    # Handle shapes robustly
                    # Expected: (1, 1, 1, 64, 100, 1) or similar
                    # We want to average over axes 1 and 3 (rx_ant, tx_ant)
                    # And squeeze axis 0, 2, 5
                    mag = np.mean(np.abs(a_val), axis=(1, 3)) # Result: (1, 1, paths, 1) -> likely (1,1,100,1)
                    mag = mag.flatten() # Should be (100,)
                except Exception as e:
                    print(f"Error processing sample {i} for CSV: {e}")
                    continue
                
                # Delays: tau_val shape (1, 1, paths) -> flatten
                delays = tau_val.flatten()
                
                # Sort by magnitude (descending) to get dominant paths
                # Although they might be sorted by delay usually, let's just take first N if they are ordered by power?
                # Sionna usually returns paths sorted by...? Often not sorted by power.
                # Let's sort manually to identify "dominant" ones.
                
                # Get indices of top N magnitudes
                # Note: -1 means invalid path in tau, magnitude should be 0 there.
                # valid_mask = delays >= 0
                
                sorted_indices = np.argsort(mag)[::-1] # Descending order
                
                row = {
                    'sample_id': i,
                    'pos_x': pos[0],
                    'pos_y': pos[1],
                    'pos_z': pos[2]
                }
                
                count = 0
                for idx in sorted_indices:
                    if count >= NUM_CSV_PATHS:
                        break
                    # Check if valid path (delay != -1)
                    if delays[idx] < 0:
                        continue
                        
                    row[f'path_{count}_mag'] = mag[idx]
                    row[f'path_{count}_delay'] = delays[idx]
                    count += 1
                
                # Fill remaining with 0/-1 if not enough valid paths
                while count < NUM_CSV_PATHS:
                    row[f'path_{count}_mag'] = 0.0
                    row[f'path_{count}_delay'] = -1.0
                    count += 1
                    
                writer.writerow(row)
        print("CSV saved successfully.")
    except Exception as e:
        print(f"Failed to save CSV: {e}")

    if args.preview: scene.preview()
    try: os.remove(scene_filename)
    except: pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Sionna Smart Factory Dataset")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--max_depth", type=int, default=5, help="Ray tracing max depth (overrides config if provided)")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU ID (overrides config if >= 0)")
    parser.add_argument("--output_filename", type=str, default="factory_dataset_refactored.h5", help="Output filename")
    parser.add_argument("--preview", action="store_true", help="Preview scene")
    args = parser.parse_args(); generate_factory_dataset(args)
