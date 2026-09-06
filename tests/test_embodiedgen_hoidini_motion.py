import numpy as np
import pytest

from nursery.humanoid.embodiedgen_hoidini.run_motion import alignment_transform
from nursery.humanoid.embodiedgen_hoidini import BridgeValidationError


def test_prefix_alignment_keeps_feet_grounded_and_body_rigid():
    source_object = np.array([2.0, 3.0, 0.9])
    target_object = np.array([4.0, 1.0, 1.1])
    transform = alignment_transform(source_object, target_object, 90)
    aligned_object = transform[:3, :3] @ source_object + transform[:3, 3]
    np.testing.assert_allclose(aligned_object, [4, 1, 0.9])
    # Object height is supplied separately from the scene; raising the table
    # must not also raise the human's feet.
    body = np.array([[2, 2, 0], [2, 2.2, 0], [2, 2.1, 1.6]])
    aligned_body = body @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(aligned_body[:, 2], body[:, 2])
    np.testing.assert_allclose(np.linalg.norm(aligned_body[2] - aligned_body[0]),
                               np.linalg.norm(body[2] - body[0]))
    np.testing.assert_allclose(np.linalg.det(transform[:3, :3]), 1)


@pytest.mark.parametrize("source,target,yaw", [
    ([0, 0], [1, 2, 3], 0),
    ([0, 0, np.nan], [1, 2, 3], 0),
    ([0, 0, 0], [1, 2, 3], np.inf),
])
def test_invalid_prefix_alignment_fails(source, target, yaw):
    with pytest.raises(BridgeValidationError):
        alignment_transform(source, target, yaw)
