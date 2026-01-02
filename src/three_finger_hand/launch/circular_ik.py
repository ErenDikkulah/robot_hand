import numpy as np
import ikpy.chain
import os
import subprocess
from ament_index_python.packages import get_package_share_directory

def run_circular_ik():
    # Manually generate temporary URDF with debug info
    try:
        pkg_name = 'three_finger_hand'
        pkg_share = get_package_share_directory(pkg_name)
        xacro_file = os.path.join(pkg_share, 'urdf', 'three_finger_hand.urdf.xacro')
        urdf_output = "/tmp/circular_hand_debug.urdf"

        subprocess.run(['xacro', xacro_file, '-o', urdf_output], check=True)
        print(f"Temporary URDF generated at {urdf_output}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating URDF: {e}")
        return

    # Load URDF and Waypoints
    urdf_path = "/tmp/circular_hand_debug.urdf"
    waypoints = np.load('src/waypoints.npy')

    # Load all chains
    chains = {
        'f1': ikpy.chain.Chain.from_urdf_file(urdf_path, base_elements=["palm"], name="finger_1"),
        'f2': ikpy.chain.Chain.from_urdf_file(urdf_path, base_elements=["palm"], name="finger_2"),
        'f3': ikpy.chain.Chain.from_urdf_file(urdf_path, base_elements=["palm"], name="finger_3")
    }

    # Mask chains to only move finger joints
    for c in chains.values():
        mask = [False] * len(c.links)
        for i in range(1, len(c.links) - 1): mask[i] = True
        c.active_links_mask = mask

    for finger, chain in chains.items():
        n_links = len(chain.links)
        print(f"\n{finger.upper()} Chain Info:")
        print(f"Chain Structure: {[link.name for link in chain.links]}")
        print(f"Active Joint Mask: {chain.active_links_mask}")

    trajectory_12dof = []
    # last_ik_res = [0.0] * n_links
    success_count = 0
    last_results = {k: [0.0] * len(v.links) for k, v in chains.items()}

    print("\nCalculation started...")
    for i, target in enumerate(waypoints):
        full_cmd = [0.0] * 12

        # IK Solution
        for f_idx, (key, chain) in enumerate(chains.items()):
            ik_res = chain.inverse_kinematics(target, initial_position=last_results[key])
            last_results[key] = ik_res

            # Verification with FK
            computed_pos = chain.forward_kinematics(ik_res)[:3, 3]
            dist = np.linalg.norm(target - computed_pos)

            # Error analysis for the first point
            if i == 0:
                print(f"First Point Analysis for {key}:")
                print(f"  Target: {target}")
                print(f"  Reached: {computed_pos}")
                print(f"  Distance Error: {dist:.4f}m")

            # 1 cm tolerance
            if dist < 0.01:
                success_count += 1
                last_results[key] = ik_res

            # 12-DOF Packaging
            start = f_idx * 4
            full_cmd[start: start + 4] = ik_res[1:5].tolist()
        trajectory_12dof.append(full_cmd)

    success_rate = (success_count / (len(waypoints) * 3)) * 100
    print(f"\nCompleted: Success Rate %{success_rate:.2f}")

    if success_count > 0:
        np.save('src/joint_trajectory.npy', np.array(trajectory_12dof))
        print("Trajectory saved.")
if __name__ == "__main__":
    run_circular_ik()
