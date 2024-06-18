import numpy as np
import numpy.typing as npt
from collections import Counter

class PointsGrid:
    def __init__(self, grid_size: int = 10) -> None:
        self.grid_size = grid_size

        # Mask has shape (home goals, away goals, pred home goals, pred away goals) = points for pred
        self.mask: npt.NDArray[np.int32] = np.fromfunction(
            function=lambda home, away, i, j: np.abs(home-i) + np.abs(away-j),
            shape=(grid_size, grid_size, grid_size, grid_size),
        )

    @staticmethod
    def tilt_array(arr: npt.NDArray[np.number], alpha: float = 0, *, renormalize: bool = True) -> npt.NDArray[np.float64]:
        """
        Static method that tilts an array by some scaling factor alpha
        As alpha -> inf, the array gets skewed towards the origin
        As alpha -> -inf, the array gets skewed towards the highest index-value
        With alpha = 0 there is no change
        """
        assert len(arr.shape) == 2
        weights = np.fromfunction(
            function=lambda i, j: (2 / ((i+1)**alpha + (j+1)**alpha)),
            shape=arr.shape,
        )

        tilted_arr = np.multiply(arr, weights).astype(np.float64)

        # The expected points get scaled during the tilt
        # We can recover the original size of the expected points by renormalizing here
        if renormalize:
            return tilted_arr / (tilted_arr.sum(axis=None) / arr.sum(axis=None))
        return tilted_arr
    
    @staticmethod
    def normalize(arr: npt.NDArray[np.number]) -> npt.NDArray[np.float64]:
        return (arr/arr.sum(axis=None)).astype(np.float64)

    def generate_score_grid(
        self, 
        samples: npt.NDArray[np.int32],
        *,
        filter_samples: bool = True,
    ) -> npt.NDArray[np.int32]:
        score_samples = samples
        if filter_samples:
            score_samples = score_samples[~np.any(score_samples >= self.grid_size, axis=1)]

        counts: Counter[tuple[int, int]] = Counter(tuple(s) for s in score_samples)
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        for (i, j), count in counts.items():
            grid[i, j] = count

        return grid
    
    def generate_expected_points_grid(
        self, 
        samples: npt.NDArray[np.int32],
        *,
        filter_samples: bool = True,
        apply_tilt: bool = False,
        alpha: float = 0,
        renormalize_tilt: bool = True,
    ) -> npt.NDArray[np.float64]:
        score_grid = self.generate_score_grid(samples=samples, filter_samples=filter_samples)

        expected_points = np.multiply(self.mask, score_grid).sum(axis=(2,3), dtype=np.float64)/samples.shape[0]

        if apply_tilt:
            return PointsGrid.tilt_array(arr=expected_points, alpha=alpha, renormalize=renormalize_tilt)
        return expected_points
    
    def get_score_prediction(
        self,
        samples: npt.NDArray[np.int32],
        *,
        apply_tilt: bool = True,
        alpha: float = 0,
    ) -> tuple[int, int]:
        exp_points_grid = self.generate_expected_points_grid(
            samples=samples,
            filter_samples=True,
            apply_tilt=apply_tilt,
            alpha=alpha,
            renormalize_tilt=True,
        )

        pred = np.unravel_index(
            indices=np.argmin(a=exp_points_grid, axis=None),
            shape=exp_points_grid.shape,
        )
        return pred
