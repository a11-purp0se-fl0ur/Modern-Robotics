import numpy as np

def inverse_kinematics(n_joints, joint_lengths, desired_pos, initial_angles_deg, max_iterations=50, tolerance=1e-3):
    # Convert initial angles from degrees to radians
    theta_rad = np.radians(initial_angles_deg)

    # Initialization
    error = 1E6
    it = 0

    # Print header
    print('\nSTART OF ALGORITHM')
    print_header = ('iter', 'theta (deg)', '(x,y)', 'error')
    print('{:5} {:>15} {:>25} {:>10}'.format(*print_header))

    while it <= max_iterations and error > tolerance:
        J = np.zeros((2, n_joints))
        x = y = 0

        # Build the Jacobian and calculate current position
        for i in range(n_joints):
            angle_sum = sum(theta_rad[:i+1])
            x += joint_lengths[i] * np.cos(angle_sum)
            y += joint_lengths[i] * np.sin(angle_sum)

            for j in range(i, n_joints):
                J[0, j] -= joint_lengths[i] * np.sin(sum(theta_rad[:j+1])) # x-component
                J[1, j] += joint_lengths[i] * np.cos(sum(theta_rad[:j+1])) # y-component

        # Current end-effector position
        current_pos = np.array([x, y])
        diff = desired_pos - current_pos
        error = np.linalg.norm(diff)

        # Print current iteration
        ang_print = np.array2string(np.rad2deg(theta_rad), formatter={'float_kind': '{:<8.2f}'.format})
        xyz_print = np.array2string(current_pos, formatter={'float_kind': '{:<10.2e}'.format})
        errors_print = np.array2string(np.array([error]), formatter={'float_kind': '{:<6.2e}'.format})
        print('{:<5d} {} {} {}'.format(it, ang_print, xyz_print, errors_print))

        # Update theta
        J_pinv = np.linalg.pinv(J)  # Pseudo-inverse of Jacobian
        theta_rad += J_pinv @ diff

        # Normalize angles to be within [0, 2*pi]
        theta_rad = theta_rad % (2 * np.pi)

        it += 1

    return np.rad2deg(theta_rad), current_pos

# Example usage:
n_joints = 2
joint_lengths = [1, 1]
desired_pos = [0.366, 1.366]
initial_angles_deg = [10, 10]

final_angles, final_position = inverse_kinematics(n_joints, joint_lengths, desired_pos, initial_angles_deg)
print('\nFinal joint angles (degrees):', final_angles)
print('Final end-effector position:', final_position)