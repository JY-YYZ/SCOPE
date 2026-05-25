import torch


class ActivationCovariance:
    """Online inverse second-moment estimator with a Woodbury update."""

    def __init__(
        self,
        hidden_dim: int,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        ridge: float = 1e-4,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self.dtype = dtype
        self.ridge = ridge
        # Store the inverse matrix in float64; eigendecomposition is sensitive here.
        self.cov_inv = torch.eye(hidden_dim, device=self.device, dtype=torch.float64) / ridge
        self.n_samples = 0

    @torch.no_grad()
    def update(self, activations: torch.Tensor, chunk_size: int = 512) -> None:
        """Update (ridge * I + X X^T)^-1 from activations shaped [..., hidden_dim]."""
        x_all = activations.reshape(-1, self.hidden_dim).to(self.device, dtype=torch.float64)

        for start in range(0, x_all.shape[0], chunk_size):
            # X is arranged as [hidden_dim, samples] to match the Woodbury formula.
            x = x_all[start : start + chunk_size].T
            if x.numel() == 0:
                continue

            cov_inv_x = self.cov_inv @ x
            core = torch.eye(x.shape[1], device=self.device, dtype=torch.float64) + x.T @ cov_inv_x
            # Symmetrization and jitter reduce numerical drift across many updates.
            core = (core + core.T) / 2
            core = core + 1e-10 * torch.eye(core.shape[0], device=self.device, dtype=torch.float64)

            solved = torch.linalg.solve(core, cov_inv_x.T)
            self.cov_inv -= cov_inv_x @ solved
            self.cov_inv = (self.cov_inv + self.cov_inv.T) / 2
            self.n_samples += x.shape[1]

    @torch.no_grad()
    def null_space_projection(
        self,
        energy_threshold: float = 0.95,
        min_null_dim: int = 0,
        max_protected_dim: int | None = None,
    ) -> torch.Tensor:
        """Return I - U U^T, where U spans the high-energy activation subspace."""
        matrix = (self.cov_inv + self.cov_inv.T) / 2
        evals, evecs = torch.linalg.eigh(matrix)

        # Eigenvalues of the covariance are inverse eigenvalues of cov_inv.
        cov_evals = 1.0 / torch.clamp(evals, min=1e-15)
        order = torch.argsort(cov_evals, descending=True)
        energy = torch.cumsum(cov_evals[order], dim=0) / cov_evals.sum().clamp_min(1e-15)

        # The protected subspace keeps high-energy general/expert directions.
        protected_dim = int(torch.searchsorted(energy, energy_threshold).item() + 1)
        if max_protected_dim is not None:
            protected_dim = min(protected_dim, max_protected_dim)
        if min_null_dim > 0:
            protected_dim = min(protected_dim, self.hidden_dim - min_null_dim)
        protected_dim = max(1, min(protected_dim, self.hidden_dim))

        basis = evecs[:, order[:protected_dim]].to(self.dtype)
        eye = torch.eye(self.hidden_dim, device=self.device, dtype=self.dtype)
        return eye - basis @ basis.T
