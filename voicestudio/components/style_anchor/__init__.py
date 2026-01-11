import torch
from torch import nn


class DirectStyleAnchorEmbedding(nn.Embedding):
    """
    Style Embedding Anchor using Direct Optimization.

    This class extends nn.Embedding to enable selective tuning of specific token embeddings
    (e.g., BOS token) while keeping others frozen. It's a drop-in replacement for nn.Embedding
    that adds style anchor functionality.

    Args:
        num_embeddings (int): Size of the vocabulary
        embedding_dim (int): Dimension of embeddings
        anchor_token_id (int or tuple[int]): Token ID(s) to use as style anchor
        pretrained_weight (torch.Tensor, optional): Pretrained embedding weights to initialize from
        padding_idx (int, optional): Padding index (passed to nn.Embedding)
        **kwargs: Additional arguments passed to nn.Embedding

    Example:
        >>> # Replace existing embedding layer
        >>> embedding = DirectStyleAnchorEmbedding(50257, 768, anchor_token_id=1)  # BOS token = 1
        >>> # Use like normal nn.Embedding
        >>> output = embedding(input_ids)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        anchor_token_id: int | tuple[int] = 1,
        pretrained_weight: torch.Tensor | None = None,
        padding_idx: int | None = None,
        **kwargs
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, **kwargs)

        # Handle single or multiple anchor tokens
        if isinstance(anchor_token_id, int):
            self.anchor_token_ids = [anchor_token_id]
        else:
            self.anchor_token_ids = list(anchor_token_id)

        # Initialize with pretrained weights if provided
        if pretrained_weight is not None:
            with torch.no_grad():
                self.weight.copy_(pretrained_weight)

        # Create learnable delta (residual) parameters for each anchor token
        # These will be ADDED to the pretrained embeddings during forward pass
        # Initialize to zeros so initial behavior matches pretrained model
        self.anchor_deltas = nn.ParameterList([
            nn.Parameter(torch.zeros(embedding_dim))
            for _ in self.anchor_token_ids
        ])

        # Freeze the main embedding weight
        self.weight.requires_grad = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that adds learned deltas to anchor token embeddings.

        This additive approach preserves pretrained knowledge while allowing
        task-specific adaptation with minimal parameters.

        Args:
            input: Input tensor of token IDs (any shape)

        Returns:
            Embedding tensor with anchor deltas applied
        """
        # Get base embeddings from pretrained weights
        embeddings = super().forward(input)

        # Add learned deltas to anchor token embeddings
        for anchor_id, delta in zip(self.anchor_token_ids, self.anchor_deltas):
            # Find positions where anchor token appears
            mask = (input == anchor_id)
            if mask.any():
                # Add delta to those positions (preserves pretrained knowledge)
                embeddings = embeddings + mask.unsqueeze(-1) * delta

        return embeddings

    def get_anchor_embeddings(self) -> list[torch.Tensor]:
        """
        Get the current effective anchor embeddings (pretrained + delta).

        Returns:
            List of effective embeddings for each anchor token
        """
        return [
            self.weight[anchor_id] + delta
            for anchor_id, delta in zip(self.anchor_token_ids, self.anchor_deltas)
        ]

    def get_anchor_deltas(self) -> list[torch.Tensor]:
        """Get the learned delta values for anchor tokens."""
        return [delta.clone() for delta in self.anchor_deltas]

    def reset_anchor_embeddings(self, token_id: int | None = None):
        """
        Reset anchor delta(s) to zero (returning to pretrained values).

        Args:
            token_id: Specific token ID to reset, or None to reset all
        """
        with torch.no_grad():
            if token_id is None:
                # Reset all deltas to zero
                for delta in self.anchor_deltas:
                    delta.zero_()
            else:
                # Reset specific delta
                idx = self.anchor_token_ids.index(token_id)
                self.anchor_deltas[idx].zero_()

    def get_tunable_parameters(self) -> int:
        """Get total number of tunable parameters."""
        return sum(p.numel() for p in self.anchor_deltas)

    def get_embedding_stats(self) -> dict:
        """
        Get statistics about anchor embeddings for debugging.

        Returns:
            Dictionary with stats for each anchor token including delta norms,
            mean/std values, and effective embedding norms
        """
        stats = {}
        for i, (anchor_id, delta) in enumerate(
            zip(self.anchor_token_ids, self.anchor_deltas)
        ):
            effective = self.weight[anchor_id] + delta
            stats[f'anchor_{i}'] = {
                'token_id': anchor_id,
                'delta_norm': delta.norm().item(),
                'delta_mean': delta.mean().item(),
                'delta_std': delta.std().item(),
                'effective_norm': effective.norm().item(),
            }
        return stats



class EncoderStyleAnchorEmbedding(nn.Embedding):
    """
    Style Embedding Anchor using Indirect Optimization with 2-Layer MLP (P-Tuning concept).

    This class extends nn.Embedding and uses a small MLP to generate style anchor embeddings
    from learnable base parameters. This provides more flexibility than direct optimization.

    Args:
        num_embeddings (int): Size of the vocabulary
        embedding_dim (int): Dimension of embeddings
        anchor_token_id (int or tuple[int]): Token ID(s) to use as style anchor
        pretrained_weight (torch.Tensor, optional): Pretrained embedding weights
        hidden_dim (int, optional): Hidden dimension of the MLP encoder (default: embedding_dim)
        padding_idx (int, optional): Padding index
        **kwargs: Additional arguments for nn.Embedding

    Example:
        >>> # More expressive style anchor with MLP
        >>> embedding = EncoderStyleAnchorEmbedding(50257, 768, anchor_token_id=1)
        >>> output = embedding(input_ids)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        anchor_token_id: int | tuple[int] = 1,
        pretrained_weight: torch.Tensor | None = None,
        hidden_dim: int | None = None,
        padding_idx: int | None = None,
        **kwargs
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, **kwargs)

        # Handle single or multiple anchor tokens
        if isinstance(anchor_token_id, int):
            self.anchor_token_ids = [anchor_token_id]
        else:
            self.anchor_token_ids = list(anchor_token_id)

        # Initialize with pretrained weights if provided
        if pretrained_weight is not None:
            with torch.no_grad():
                self.weight.copy_(pretrained_weight)

        # Hidden dimension defaults to embedding_dim
        if hidden_dim is None:
            hidden_dim = embedding_dim

        # Create learnable base parameters and MLP encoders for each anchor
        # MLP generates delta (residual) to add to pretrained embedding
        # Initialize bases to small values so initial delta is near zero
        self.anchor_bases = nn.ParameterList([
            nn.Parameter(torch.randn(embedding_dim) * 0.01)
            for _ in self.anchor_token_ids
        ])

        self.anchor_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, embedding_dim)
            )
            for _ in self.anchor_token_ids
        ])

        # Freeze the main embedding weight
        self.weight.requires_grad = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that generates and adds style anchor deltas via MLP.

        The MLP generates a delta that is added to the pretrained embedding,
        preserving pretrained knowledge while enabling expressive adaptation.

        Args:
            input: Input tensor of token IDs

        Returns:
            Embedding tensor with MLP-generated deltas added
        """
        # Get base embeddings from pretrained weights
        embeddings = super().forward(input)

        # Generate and add deltas for each anchor token
        for anchor_id, base, encoder in zip(
                self.anchor_token_ids, self.anchor_bases, self.anchor_encoders
        ):
            # Generate delta from base via MLP
            delta = encoder(base)

            # Find positions where anchor token appears
            mask = (input == anchor_id)
            if mask.any():
                # Add delta to those positions
                embeddings = embeddings + mask.unsqueeze(-1) * delta

        return embeddings

    def get_anchor_embeddings(self) -> list[torch.Tensor]:
        """
        Get the current effective anchor embeddings (pretrained + generated delta).

        Returns:
            List of effective embeddings for each anchor token
        """
        return [
            self.weight[anchor_id] + encoder(base)
            for anchor_id, base, encoder in zip(
                self.anchor_token_ids, self.anchor_bases, self.anchor_encoders
            )
        ]

    def get_anchor_deltas(self) -> list[torch.Tensor]:
        """Get the MLP-generated delta values for anchor tokens."""
        return [
            encoder(base).clone()
            for base, encoder in zip(self.anchor_bases, self.anchor_encoders)
        ]

    def get_tunable_parameters(self) -> int:
        """Get total number of tunable parameters."""
        return sum(p.numel() for p in self.anchor_bases) + \
               sum(p.numel() for p in self.anchor_encoders.parameters())

    def get_embedding_stats(self) -> dict:
        """
        Get statistics about anchor embeddings for debugging.

        Returns:
            Dictionary with stats for each anchor token including base stats,
            delta norms, and effective embedding norms
        """
        stats = {}
        for i, (anchor_id, base, encoder) in enumerate(
            zip(self.anchor_token_ids, self.anchor_bases, self.anchor_encoders)
        ):
            delta = encoder(base)
            effective = self.weight[anchor_id] + delta
            stats[f'anchor_{i}'] = {
                'token_id': anchor_id,
                'base_norm': base.norm().item(),
                'delta_norm': delta.norm().item(),
                'delta_mean': delta.mean().item(),
                'delta_std': delta.std().item(),
                'effective_norm': effective.norm().item(),
            }
        return stats
