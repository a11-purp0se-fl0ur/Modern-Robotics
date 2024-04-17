import numpy as np
import pytest
from Functions.Mia_Functions import *


def test_PoE_identity():
    # Test PoE function with zero joint angles
    M = np.eye(4)  # Home configuration as identity
    B = np.zeros((6, 1))  # No movement
    theta = np.array([0])  # Zero angle
    expected = np.eye(4)  # Expected result is also identity
    assert np.allclose(PoE_Body(theta, M, B), expected), "PoE with zero angles should return home configuration"


def test_PoE_known_configuration():
    # Test PoE function with a known configuration
    Rsb = np.eye(3)
    p = np.array([2, 0, 0])
    M = constructT(Rsb, p)

    S = np.array([[0, 0],
                  [0, 0],
                  [1, 1],
                  [0, 0],
                  [0, -1],
                  [0, 0]])
    B = adjoint(np.linalg.inv(M)) @ S

    theta_deg = np.array([30, 90])  # Joint angles
    theta = np.deg2rad(theta_deg)

    expected = np.array([[-0.5, -0.866, 0, 0.366],
                         [0.866, -0.5, 0, 1.366],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])  # Expected transformation matrix

    assert np.allclose(PoE_Body(theta, M, B), expected, rtol=1e-4, atol=1e-4), "PoE does not match expected output for known configuration"
