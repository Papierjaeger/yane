"""Convolutional NEAT — CoDeepNEAT-inspired image processing front-end.

Adds an optional conv stack to any YANE genome so it can process 2-D image
inputs without modifying the core ``NodeType`` enum or ``genome.forward()``.

Architecture
------------
::

    [H × W × C pixels]
          ↓
    ConvBlock 0   (K₀×K₀, in_ch=C,  out_ch=F₀)
          ↓  global-average-pool
    [F₀ feature values]
          ↓
    ConvBlock 1   (K₁×K₁, in_ch=F₀, out_ch=F₁)
          ↓  global-average-pool
    [F₁ feature values]
          ↓  ... (stack of ConvBlocks)
          ↓
    [F_total flat vector]  (F_total = sum of each block's out_channels)
          ↓
    genome.forward(flat_vector)   ← existing NEAT forward
          ↓
    [n_outputs]

Weight-sharing guarantee
------------------------
A K×K kernel with *in* input channels and *out* output channels has exactly
``K * K * in`` learned weights per output channel, regardless of the spatial
size of the feature map it is applied to.  This is asserted in the
:class:`ConvBlock` constructor.

Integration
-----------
1. Call ``yane.set_conv_neat(blocks=[...])`` or use
   ``set_conv_neat(n_blocks=2, kernel_size=3, out_channels=8)``.
2. Call ``yane.configure(n_inputs=yane.conv_n_inputs(), n_outputs=...)``
   (or pass ``configure`` the correct *n_inputs* yourself).
3. Use ``genome.forward_image(pixels, h, w, c)`` in the evaluator instead
   of ``genome.forward(flat_inputs)``.

Zero-cost when disabled: ``genome.conv_blocks = None`` means no overhead.
"""
from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome


# ---------------------------------------------------------------------------
# ConvBlock
# ---------------------------------------------------------------------------

@dataclass
class ConvBlock:
    """One weight-shared convolutional layer.

    Each output channel has its own set of kernel weights that are shared
    across all spatial positions of the feature map.

    Parameters
    ----------
    kernel_size :
        Width (= height) of the square convolution kernel.
    stride :
        Step size between successive filter applications.
    in_channels :
        Number of input feature channels.
    out_channels :
        Number of output feature channels.
    activation :
        Activation function name (any YANE-supported name, e.g. ``"relu"``).
    kernels :
        ``out_channels`` weight vectors, each of length
        ``kernel_size * kernel_size * in_channels``.
        Automatically created if *None*.
    biases :
        Per-output-channel bias.  Automatically created if *None*.
    """

    kernel_size: int = 3
    stride: int = 1
    in_channels: int = 1
    out_channels: int = 4
    activation: str = "relu"
    kernels: list[list[float]] = field(default_factory=list)
    biases: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        n_weights = self.kernel_size * self.kernel_size * self.in_channels
        if not self.kernels:
            # Xavier-like init
            scale = math.sqrt(2.0 / max(1, n_weights))
            self.kernels = [
                [random.gauss(0.0, scale) for _ in range(n_weights)]
                for _ in range(self.out_channels)
            ]
        if not self.biases:
            self.biases = [0.0] * self.out_channels

        # Invariant: shared kernel size must not depend on image size
        assert all(len(k) == n_weights for k in self.kernels), (
            f"ConvBlock: each kernel must have {n_weights} weights "
            f"(kernel_size={self.kernel_size} × in_channels={self.in_channels})"
        )

    @property
    def n_weights_per_channel(self) -> int:
        """Kernel footprint per output channel (K² × in_channels).

        This count is **independent of the spatial image size** — that is the
        core weight-sharing invariant of a convolutional layer.
        """
        return self.kernel_size * self.kernel_size * self.in_channels

    @property
    def total_params(self) -> int:
        """Total learnable parameters (weights + biases)."""
        return self.n_weights_per_channel * self.out_channels + self.out_channels

    def copy(self) -> ConvBlock:
        return ConvBlock(
            kernel_size=self.kernel_size,
            stride=self.stride,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            activation=self.activation,
            kernels=[list(k) for k in self.kernels],
            biases=list(self.biases),
        )

    def mutate(self, sigma: float = 0.1, rng: random.Random | None = None) -> None:
        """Perturb kernel weights and biases with Gaussian noise."""
        _rng = rng or random
        for k in self.kernels:
            for i in range(len(k)):
                k[i] += _rng.gauss(0.0, sigma)
        for i in range(len(self.biases)):
            self.biases[i] += _rng.gauss(0.0, sigma * 0.5)

    def crossover(self, other: ConvBlock) -> ConvBlock:
        """Uniform per-channel crossover; inherits structure from *self*."""
        child = self.copy()
        for c in range(min(len(child.kernels), len(other.kernels))):
            if random.random() < 0.5:
                child.kernels[c] = list(other.kernels[c])
                child.biases[c] = other.biases[c]
        return child

    # ------------------------------------------------------------------
    # Forward pass (applies to one image plane)
    # ------------------------------------------------------------------

    def forward(
        self,
        feature_maps: list[list[float]],
        h: int,
        w: int,
    ) -> list[float]:
        """Apply this block to *feature_maps* and return global-average-pooled outputs.

        Parameters
        ----------
        feature_maps :
            ``in_channels`` flattened planes, each of length ``h * w``.
        h, w :
            Spatial dimensions of each plane.

        Returns
        -------
        list[float]
            ``out_channels`` values (one per output channel, after global
            average pooling).  Length is always ``out_channels``, regardless
            of the spatial dimensions — this is the image-size-independence
            guarantee.
        """
        assert len(feature_maps) == self.in_channels, (
            f"ConvBlock.forward: expected {self.in_channels} input planes, "
            f"got {len(feature_maps)}"
        )
        k = self.kernel_size
        s = self.stride
        h_out = max(1, (h - k) // s + 1)
        w_out = max(1, (w - k) // s + 1)
        act = _activation(self.activation)

        result: list[float] = []
        for out_c in range(self.out_channels):
            kern = self.kernels[out_c]
            bias = self.biases[out_c]
            total = 0.0
            n_positions = 0
            for row in range(h_out):
                for col in range(w_out):
                    val = bias
                    wi = 0
                    for in_c in range(self.in_channels):
                        plane = feature_maps[in_c]
                        for kr in range(k):
                            for kc in range(k):
                                r = row * s + kr
                                c = col * s + kc
                                pix = plane[r * w + c] if (0 <= r < h and 0 <= c < w) else 0.0
                                val += kern[wi] * pix
                                wi += 1
                    total += act(val)
                    n_positions += 1
            # Global average pool over spatial positions
            result.append(total / max(1, n_positions))
        return result


# ---------------------------------------------------------------------------
# ConvStack — ordered list of ConvBlocks
# ---------------------------------------------------------------------------

class ConvStack:
    """An evolvable stack of :class:`ConvBlock` layers.

    The first block's ``in_channels`` must match the image's channel count.
    Subsequent blocks chain: ``block[i+1].in_channels == block[i].out_channels``.

    Parameters
    ----------
    blocks :
        Ordered list of :class:`ConvBlock` objects.
    """

    def __init__(self, blocks: list[ConvBlock]) -> None:
        self.blocks: list[ConvBlock] = list(blocks)

    @property
    def n_outputs(self) -> int:
        """Flat output dimension after global-average-pooling all blocks.

        Equal to ``sum(b.out_channels for b in blocks)``.  This is the
        ``n_inputs`` value to pass to ``NeuroEvolution.configure()``.
        """
        return sum(b.out_channels for b in self.blocks)

    def forward_image(
        self,
        pixels: list[float],
        height: int,
        width: int,
        channels: int,
    ) -> list[float]:
        """Process a flat image and return a flat feature vector.

        Parameters
        ----------
        pixels :
            Flat image data of length ``height * width * channels``.
            Expected channel-last ordering: pixel (r, c, ch) is at index
            ``r * width * channels + c * channels + ch``.
        height, width :
            Spatial dimensions.
        channels :
            Number of image channels (1 = grayscale, 3 = RGB).

        Returns
        -------
        list[float]
            Flat feature vector of length :attr:`n_outputs` (= sum of
            all ``out_channels``).  Always the same length regardless of
            ``height`` and ``width``.
        """
        # Split pixels into per-channel planes (channel-last → planes)
        planes: list[list[float]] = []
        for ch in range(channels):
            plane = [
                pixels[r * width * channels + c * channels + ch]
                for r in range(height)
                for c in range(width)
            ]
            planes.append(plane)

        h, w = height, width
        features: list[float] = []
        for block in self.blocks:
            # Block expects its in_channels planes
            assert len(planes) == block.in_channels, (
                f"Block in_channels mismatch: block.in_channels={block.in_channels}, "
                f"but received {len(planes)} planes"
            )
            block_out = block.forward(planes, h, w)
            features.extend(block_out)
            # For the next block: produce new planes from this block's outputs.
            # We store each output value as a 1×1 "plane" (already pooled).
            # The next block receives `out_channels` planes, each 1×1.
            h_next, w_next = 1, 1
            planes = [[v] for v in block_out]
            h, w = h_next, w_next

        return features

    def copy(self) -> ConvStack:
        return ConvStack([b.copy() for b in self.blocks])

    def crossover(self, other: ConvStack) -> ConvStack:
        """Uniform per-block crossover; extra blocks from *self* are kept."""
        child_blocks = []
        for i, blk_a in enumerate(self.blocks):
            if i < len(other.blocks):
                child_blocks.append(blk_a.crossover(other.blocks[i]))
            else:
                child_blocks.append(blk_a.copy())
        return ConvStack(child_blocks)


# ---------------------------------------------------------------------------
# Mutation helpers
# ---------------------------------------------------------------------------

def add_conv_block(
    stack: ConvStack,
    kernel_size: int = 3,
    stride: int = 1,
    out_channels: int = 4,
    activation: str = "relu",
    rng: random.Random | None = None,
) -> None:
    """Append a new :class:`ConvBlock` to *stack* in-place.

    The new block's ``in_channels`` is automatically set to the current
    stack's last block's ``out_channels`` (or 1 if the stack is empty).
    """
    in_ch = stack.blocks[-1].out_channels if stack.blocks else 1
    new_block = ConvBlock(
        kernel_size=kernel_size,
        stride=stride,
        in_channels=in_ch,
        out_channels=out_channels,
        activation=activation,
    )
    if rng is not None:
        new_block.mutate(sigma=0.3, rng=rng)
    stack.blocks.append(new_block)


def mutate_conv_stack(
    stack: ConvStack,
    sigma: float = 0.1,
    add_block_prob: float = 0.05,
    rng: random.Random | None = None,
) -> None:
    """Mutate all blocks in *stack* and optionally add a new one."""
    _rng = rng or random
    for block in stack.blocks:
        block.mutate(sigma=sigma, rng=_rng)
    if _rng.random() < add_block_prob:
        add_conv_block(stack, rng=_rng)


# ---------------------------------------------------------------------------
# Activation functions (pure Python, no YANE dependency)
# ---------------------------------------------------------------------------

def _activation(name: str):
    if name == "relu":
        return lambda v: max(0.0, v)
    if name == "sigmoid":
        return lambda v: 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, v))))
    if name == "tanh":
        return math.tanh
    if name == "leaky_relu":
        return lambda v: v if v > 0.0 else 0.01 * v
    if name == "linear":
        return lambda v: v
    # default: linear
    return lambda v: v


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_conv_stack(
    n_image_channels: int = 1,
    n_blocks: int = 1,
    kernel_size: int = 3,
    out_channels: int = 8,
    activation: str = "relu",
) -> ConvStack:
    """Build a fresh :class:`ConvStack` with identical-architecture blocks.

    Parameters
    ----------
    n_image_channels :
        Number of input image channels.
    n_blocks :
        Number of stacked conv blocks.
    kernel_size :
        Kernel size for every block.
    out_channels :
        Output channels for every block.
    activation :
        Activation for every block.

    Returns
    -------
    ConvStack
        A new stack whose :attr:`~ConvStack.n_outputs` equals ``out_channels``.
    """
    blocks = []
    in_ch = n_image_channels
    for _ in range(n_blocks):
        blocks.append(ConvBlock(
            kernel_size=kernel_size,
            stride=1,
            in_channels=in_ch,
            out_channels=out_channels,
            activation=activation,
        ))
        in_ch = out_channels
    return ConvStack(blocks)
