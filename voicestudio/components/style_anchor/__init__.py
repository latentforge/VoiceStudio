import torch
from torch import nn


class StyleAnchorEmbedding(nn.Embedding):
    """
    Base class for style anchor embeddings.

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

        # Freeze the main embedding weight
        self.weight.requires_grad = False

    def _compute_deltas(self) -> list[torch.Tensor]:
        """Compute the delta values for each anchor token."""
        return []

    def _reset_deltas(self):
        """Reset the learned parameters so deltas become effectively zero (or re-initialized)."""
        pass

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

        # Get deltas from subclass implementation
        deltas = self._compute_deltas()

        # Add learned deltas to anchor token embeddings
        for anchor_id, delta in zip(self.anchor_token_ids, deltas):
            # Find positions where anchor token appears
            mask = (input == anchor_id)
            if mask.any():
                # Add delta to those positions (preserves pretrained knowledge)
                embeddings = embeddings + mask.unsqueeze(-1) * delta

        return embeddings

    @property
    def anchor_embeddings(self) -> list[torch.Tensor]:
        """
        Get the current effective anchor embeddings (pretrained + delta).

        Returns:
            List of effective embeddings for each anchor token
        """
        deltas = self._compute_deltas()
        return [
            self.weight[anchor_id] + delta
            for anchor_id, delta in zip(self.anchor_token_ids, deltas)
        ]

    @property
    def anchor_embedding_deltas(self) -> list[torch.Tensor]:
        """Get the learned delta values for anchor tokens."""
        return [delta.clone() for delta in self._compute_deltas()]

    def merge_deltas(self, reset: bool = True):
        """
        Permanently merge the learned deltas into the base embedding weights.

        This is useful for inference optimization or when saving the model.
        After merging, the deltas are reset to avoid double-counting.
        Note: This changes the underlying pretrained weights.
        
        Args:
            reset (bool): Whether to reset the deltas after merging. 
                          Set to False if sharing weights/deltas across modules 
                          to avoid premature reset.
        """
        deltas = self._compute_deltas()

        with torch.no_grad():
            for anchor_id, delta in zip(self.anchor_token_ids, deltas):
                self.weight[anchor_id] += delta

        if reset:
            self._reset_deltas()

    @property
    def stats(self) -> dict:
        """
        Get statistics about anchor embeddings for debugging.

        Returns:
            Dictionary with stats for each anchor token including base stats,
            delta norms, and effective embedding norms
        """
        return {}


class DirectStyleAnchorEmbedding(StyleAnchorEmbedding):
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
        super().__init__(
            num_embeddings,
            embedding_dim,
            anchor_token_id=anchor_token_id,
            pretrained_weight=pretrained_weight,
            padding_idx=padding_idx,
            **kwargs
        )

        # Create learnable delta (residual) parameters for each anchor token
        # These will be ADDED to the pretrained embeddings during forward pass
        # Initialize to zeros so initial behavior matches pretrained model
        self.anchor_deltas = nn.ParameterList([
            nn.Parameter(torch.zeros(embedding_dim))
            for _ in self.anchor_token_ids
        ])

    def _compute_deltas(self) -> list[torch.Tensor]:
        return [delta for delta in self.anchor_deltas]

    def _reset_deltas(self):
        with torch.no_grad():
            for delta in self.anchor_deltas:
                delta.zero_()

    @property
    def stats(self) -> dict:
        stats = {}
        for i, (anchor_id, delta) in enumerate(zip(self.anchor_token_ids, self.anchor_deltas)):
            effective = self.weight[anchor_id] + delta
            stats[f'anchor_{i}'] = {
                'token_id': anchor_id,
                'delta_norm': delta.norm().item(),
                'delta_mean': delta.mean().item(),
                'delta_std': delta.std().item(),
                'effective_norm': effective.norm().item(),
            }
        return stats


class EncoderStyleAnchorEmbedding(StyleAnchorEmbedding):
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
        super().__init__(
            num_embeddings,
            embedding_dim,
            anchor_token_id=anchor_token_id,
            pretrained_weight=pretrained_weight,
            padding_idx=padding_idx,
            **kwargs
        )

        # Hidden dimension defaults to embedding_dim // 4
        if hidden_dim is None:
            hidden_dim = embedding_dim // 4

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

    def _compute_deltas(self) -> list[torch.Tensor]:
        return [
            encoder(base)
            for base, encoder in zip(self.anchor_bases, self.anchor_encoders)
        ]

    def _reset_deltas(self):
        # Re-initialize bases and encoders to ensure we start fresh (delta approx zero or random small)
        with torch.no_grad():
            for base in self.anchor_bases:
                base.data.normal_(0, 0.01)

            for encoder in self.anchor_encoders:
                for layer in encoder.modules():
                    if isinstance(layer, nn.Linear):
                        layer.reset_parameters()

    @property
    def stats(self) -> dict:
        stats = {}
        for i, (anchor_id, base, encoder) in enumerate(zip(self.anchor_token_ids, self.anchor_bases, self.anchor_encoders)):
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


class MixedStyleAnchorEmbedding(StyleAnchorEmbedding):
    """
    Style Embedding Anchor using a mix of direct and encoder optimization.

    This class combines both Direct and Encoder style anchors, allowing different tokens to be
    optimized using different strategies within the same embedding layer.

    Args:
        num_embeddings (int): Size of the vocabulary
        embedding_dim (int): Dimension of embeddings
        direct_anchor_token_id (int | tuple[int] | None): Token ID(s) to use for direct optimization
        encoder_anchor_token_id (int | tuple[int] | None): Token ID(s) to use for encoder optimization
        pretrained_weight (torch.Tensor, optional): Pretrained embedding weights
        hidden_dim (int, optional): Hidden dimension for encoder MLP (default: embedding_dim // 4)
        padding_idx (int, optional): Padding index
        **kwargs: Additional arguments for nn.Embedding

    Example:
        >>> # Use specialized strategies for different tokens
        >>> embedding = MixedStyleAnchorEmbedding(
        ...     50257, 768,
        ...     direct_anchor_token_id=1,      # BOS token: direct optimization
        ...     encoder_anchor_token_id=50256  # EOS token: encoder optimization
        ... )
        >>> output = embedding(input_ids)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        direct_anchor_token_id: int | tuple[int] | None = None,
        encoder_anchor_token_id: int | tuple[int] | None = None,
        pretrained_weight: torch.Tensor | None = None,
        hidden_dim: int | None = None,
        padding_idx: int | None = None,
        **kwargs
    ):
        # Flatten ids to lists
        self.direct_ids, self.encoder_ids = [], []
        if direct_anchor_token_id is not None:
            if isinstance(direct_anchor_token_id, int):
                self.direct_ids = [direct_anchor_token_id]
            else:
                self.direct_ids = list(direct_anchor_token_id)
        if encoder_anchor_token_id is not None:
            if isinstance(encoder_anchor_token_id, int):
                self.encoder_ids = [encoder_anchor_token_id]
            else:
                self.encoder_ids = list(encoder_anchor_token_id)

        # Combine for base init (direct first, then encoder)
        all_anchors = self.direct_ids + self.encoder_ids

        super().__init__(
            num_embeddings,
            embedding_dim,
            anchor_token_id=tuple(all_anchors),
            pretrained_weight=pretrained_weight,
            padding_idx=padding_idx,
            **kwargs
        )

        if hidden_dim is None:
            hidden_dim = embedding_dim // 4

        # --- Direct Optimization Setup ---
        self.anchor_deltas = nn.ParameterList([
            nn.Parameter(torch.zeros(embedding_dim))
            for _ in self.direct_ids
        ])
        # --- Encoder Optimization Setup ---
        self.anchor_bases = nn.ParameterList([
            nn.Parameter(torch.randn(embedding_dim) * 0.01)
            for _ in self.encoder_ids
        ])
        self.anchor_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, embedding_dim)
            )
            for _ in self.encoder_ids
        ])

    def _compute_deltas(self) -> list[torch.Tensor]:
        # Direct deltas
        direct_deltas = [d for d in self.anchor_deltas]

        # Encoder deltas
        encoder_deltas = [
            encoder(base)
            for base, encoder in zip(self.anchor_bases, self.anchor_encoders)
        ]

        # Return combined in order (directs then encoders)
        return direct_deltas + encoder_deltas

    def _reset_deltas(self):
        with torch.no_grad():
            # Reset direct
            for delta in self.anchor_deltas:
                delta.zero_()

            # Reset encoder
            for base in self.anchor_bases:
                base.data.normal_(0, 0.01)

            for encoder in self.anchor_encoders:
                for layer in encoder.modules():
                    if isinstance(layer, nn.Linear):
                        layer.reset_parameters()

    @property
    def stats(self) -> dict:
        stats = {}

        # Direct stats
        for i, (anchor_id, delta) in enumerate(zip(self.direct_ids, self.anchor_deltas)):
            effective = self.weight[anchor_id] + delta
            stats[f'direct_anchor_{i}'] = {
                'token_id': anchor_id,
                'delta_norm': delta.norm().item(),
                'delta_mean': delta.mean().item(),
                'delta_std': delta.std().item(),
                'effective_norm': effective.norm().item(),
            }

        # Encoder stats
        for i, (anchor_id, base, encoder) in enumerate(zip(self.encoder_ids, self.anchor_bases, self.anchor_encoders)):
            delta = encoder(base)
            effective = self.weight[anchor_id] + delta
            stats[f'encoder_anchor_{i}'] = {
                'token_id': anchor_id,
                'base_norm': base.norm().item(),
                'delta_norm': delta.norm().item(),
                'delta_mean': delta.mean().item(),
                'delta_std': delta.std().item(),
                'effective_norm': effective.norm().item(),
            }

        return stats
