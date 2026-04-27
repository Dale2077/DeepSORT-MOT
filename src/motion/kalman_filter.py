"""
Kalman Filter for bounding box tracking.

8-dimensional state space: [x, y, a, h, vx, vy, va, vh]
  - (x, y): bounding box center position
  - a: aspect ratio (width / height)
  - h: height
  - vx, vy, va, vh: respective velocities

Constant velocity motion model with direct observation of (x, y, a, h).
"""

import numpy as np
import scipy.linalg


CHI2_95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877}


class KalmanFilter:
    """Kalman filter for tracking bounding boxes in image space."""

    def __init__(self):
        ndim, dt = 4, 1.0

        # State transition matrix (constant velocity model)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Observation matrix
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Uncertainty weights
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        mean : ndarray
            Mean vector (8-dimensional) of the new track.
        covariance : ndarray
            Covariance matrix (8x8) of the new track.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8-dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 covariance matrix of the object state at the previous
            time step.

        Returns
        -------
        mean : ndarray
            Predicted mean vector.
        covariance : ndarray
            Predicted covariance matrix.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot([self._motion_mat, covariance, self._motion_mat.T])
            + motion_cov
        )
        return mean, covariance

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Vectorized prediction for multiple tracks.

        Parameters
        ----------
        mean : ndarray (N, 8)
        covariance : ndarray (N, 8, 8)

        Returns
        -------
        mean : ndarray (N, 8)
        covariance : ndarray (N, 8, 8)
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones(len(mean)),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones(len(mean)),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = np.array([np.diag(s) for s in sqr])

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance)
        covariance = np.einsum("ijk,lk->ijl", left, self._motion_mat) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state mean vector (8-dimensional).
        covariance : ndarray
            The state covariance matrix (8x8).

        Returns
        -------
        mean : ndarray
            Projected mean vector (4-dimensional).
        covariance : ndarray
            Projected covariance matrix (4x4).
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            [self._update_mat, covariance, self._update_mat.T]
        )
        return mean, covariance + innovation_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            Predicted state mean vector (8-dimensional).
        covariance : ndarray
            Predicted state covariance matrix (8x8).
        measurement : ndarray
            The 4-dimensional measurement vector (x, y, a, h).

        Returns
        -------
        mean : ndarray
            Updated state mean vector.
        covariance : ndarray
            Updated state covariance matrix.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            [kalman_gain, projected_cov, kalman_gain.T]
        )
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        """Compute gating distance (squared Mahalanobis) between state and measurements.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8-dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8).
        measurements : ndarray (N, 4)
            Matrix of N measurements, each in format (x, y, a, h).
        only_position : bool
            If True, only use (x, y) for distance computation.

        Returns
        -------
        squared_maha : ndarray (N,)
            Squared Mahalanobis distance for each measurement.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        cholesky_factor = np.linalg.cholesky(covariance)
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False
        )
        return np.sum(z * z, axis=0)
