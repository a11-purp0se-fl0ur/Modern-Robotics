import numpy as np
import pytest
from Functions.Mia_Functions import *

def test_single_joint_robot():
    S = np.array([[0, 0, 1, 0, 0, 0]]).T  # Screw axis for a single revolute joint along z-axis
    theta = np.array([np.pi / 2])  # 90 degrees rotation
    expected_jacobian = np.array([[0, 0, 1, 0, 0, 0]]).T  # Expected Jacobian for a single joint rotation about z-axis

    assert np.allclose(BodyJacobian(S, theta), expected_jacobian), "Jacobian does not match for single joint robot"


def test_two_joint_robot():
    # Define screw axes for a two-joint robot where both joints rotate about the z-axis
    S = np.array([[0, 1, 1],
                  [0, 0, 0],
                  [1, 0, 0],
                  [0, 0, 0],
                  [0, 0.15, 0.15],
                  [0, 0, -0.1]])
    theta = np.array([np.pi/3,np.pi/2,np.pi/3])  # Both joints at 90 degrees
    # Expected Jacobian needs to be calculated or obtained from a trusted source
    expected_jacobian = np.array([[0, 0.5, 0.5],
              [0, 0.866, 0.866],
              [1, 0, 0],
              [0, -0.13, -0.217],
              [0, 0.075, 0.125],
              [0, 0, 0]])

    assert np.allclose(BodyJacobian(S, theta), expected_jacobian, rtol=1e-4, atol=1e-4), "Jacobian does not match for two joint robot"
