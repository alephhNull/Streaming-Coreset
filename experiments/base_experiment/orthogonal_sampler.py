import numpy as np
from scipy.stats import chi


class OrthogonalSampler:
    def __init__(self, d_in: int, n_components: int, gamma: float):
        self.d_in = d_in
        self.n_components = n_components
        self.gamma = gamma

        nb_blocks = int(np.ceil(n_components / d_in))
        W_blocks = []
        for _ in range(nb_blocks):
            G = np.random.randn(d_in, d_in)
            Q, _ = np.linalg.qr(G)
            W_blocks.append(Q)

        W_ortho = np.vstack(W_blocks)[:n_components, :]
        chi_lengths = chi.rvs(df=d_in, size=n_components)
        self.W = (W_ortho * chi_lengths[:, np.newaxis]) * np.sqrt(2 * gamma)
        self.b = np.random.uniform(0, 2 * np.pi, n_components)

    def transform(self, X: np.ndarray) -> np.ndarray:
        projection = X @ self.W.T + self.b
        return np.sqrt(2.0 / self.n_components) * np.cos(projection)
