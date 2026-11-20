"""What a forward and a backward pass of the benchmark's model cost, in floating-point operations.

The unit is the multiply-accumulate, and FLOPs are twice that; every number that leaves this module
is a ratio of two costs, where the factor cancels. One inference is one forward pass of the original
model over one response: its prompt of P tokens and its T response tokens.

How a pass is counted (Kaplan et al., 2020):

- A linear map with W weights costs W for each token in the forward pass. In the backward pass it
  costs W for the gradient of its input and W for the gradient of its weights.
- A frozen map needs no gradient of its weights, and no map needs the gradient of its input below
  the lowest weight that trains.
- The operations without weights (the products of full attention over the context, the state of
  linear attention, its convolution) cost N forward and 2 N backward.

What the methods train:

    SFT                    the last decoder layers; everything below them is frozen
    GFlowNet               low-rank adapters on the projections of every layer; the model is frozen
    CTL                    the twist network, on hidden states that generation kept
    KL-DPG, JS-DPG, GDC    all of the model's weights

`CONVENTIONS` holds four ways to count a pass; the analysis reports the largest.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from posteriorbench_methods.common import errors
from posteriorbench_methods.gfn import policy as gfn_policy

FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"
LinearMaps = dict[str, tuple[int, int]]  # the projection's name -> (inputs, outputs)
# The projections of an attention block that read the layer's input.
INPUT_PROJECTIONS: dict[str, tuple[str, ...]] = {
    FULL_ATTENTION: ("q_proj", "k_proj", "v_proj"),
    LINEAR_ATTENTION: ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"),
}
Trains = Literal["nothing", "last_layers", "adapters", "twist", "all_weights"]


class ShapeError(errors.MethodsError, ValueError):
    """A model configuration that is not of the architecture this module describes."""


def _weights(maps: Mapping[str, tuple[int, int]]) -> int:
    return sum(inputs * outputs for inputs, outputs in maps.values())


@dataclasses.dataclass(frozen=True)
class Shape:
    """The sizes of a model with the architecture of Qwen3.5: decoder layers of two kinds, each an
    attention block and a feed-forward block, between an embedding and an output layer.

    layer_types: "full_attention" or "linear_attention" for each decoder layer, from the first.
    attention_heads, key_value_heads, head_dim: query heads and key-value heads of full
        attention, each of width `head_dim`.
    tied_embeddings: whether the embedding and the output layer are one matrix."""

    hidden: int
    feed_forward: int
    vocabulary: int
    layer_types: tuple[str, ...]
    attention_heads: int
    key_value_heads: int
    head_dim: int
    linear_key_heads: int
    linear_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel: int
    tied_embeddings: bool

    def __post_init__(self) -> None:
        unknown = sorted(set(self.layer_types) - {FULL_ATTENTION, LINEAR_ATTENTION})
        if not self.layer_types or unknown:
            raise ShapeError(
                f"layers are {FULL_ATTENTION!r} or {LINEAR_ATTENTION!r}, got {unknown or 'none'}"
            )

    @classmethod
    def from_config(cls, config: Any) -> Shape:
        """The shape of a transformers configuration of Qwen3.5 (of its text model, if it has a
        vision model). Raises ShapeError for another architecture."""
        text = getattr(config, "text_config", config)
        try:
            return cls(
                hidden=int(text.hidden_size),
                feed_forward=int(text.intermediate_size),
                vocabulary=int(text.vocab_size),
                layer_types=tuple(str(kind) for kind in text.layer_types),
                attention_heads=int(text.num_attention_heads),
                key_value_heads=int(text.num_key_value_heads),
                head_dim=int(text.head_dim),
                linear_key_heads=int(text.linear_num_key_heads),
                linear_value_heads=int(text.linear_num_value_heads),
                linear_key_head_dim=int(text.linear_key_head_dim),
                linear_value_head_dim=int(text.linear_value_head_dim),
                linear_conv_kernel=int(text.linear_conv_kernel_dim),
                tied_embeddings=bool(text.tie_word_embeddings),
            )
        except (AttributeError, TypeError) as error:
            raise ShapeError(
                "the FLOPs model describes the architecture of Qwen3.5 (full-attention and "
                f"linear-attention layers); this configuration has no such sizes: {error}"
            ) from error

    @property
    def _linear_widths(self) -> tuple[int, int]:
        """The widths of the keys and of the values of linear attention, over all heads."""
        return (
            self.linear_key_heads * self.linear_key_head_dim,
            self.linear_value_heads * self.linear_value_head_dim,
        )

    def attention_maps(self, layer_type: str) -> LinearMaps:
        """The linear maps of the attention block of a layer, by the name of the projection."""
        h = self.hidden
        if layer_type == FULL_ATTENTION:
            queries = self.attention_heads * self.head_dim
            keys = self.key_value_heads * self.head_dim
            return {
                "q_proj": (h, 2 * queries),  # the queries and the gate of the block's output
                "k_proj": (h, keys),
                "v_proj": (h, keys),
                "o_proj": (queries, h),
            }
        key, value = self._linear_widths
        return {
            "in_proj_qkv": (h, 2 * key + value),
            "in_proj_z": (h, value),
            "in_proj_a": (h, self.linear_value_heads),
            "in_proj_b": (h, self.linear_value_heads),
            "out_proj": (value, h),
        }

    def feed_forward_maps(self) -> LinearMaps:
        """The linear maps of the feed-forward block of a layer."""
        h, f = self.hidden, self.feed_forward
        return {"gate_proj": (h, f), "up_proj": (h, f), "down_proj": (f, h)}

    def layer_weights(self, layer_type: str) -> int:
        """The weights of one layer's linear maps: its cost for each token of a forward pass."""
        return _weights(self.attention_maps(layer_type)) + _weights(self.feed_forward_maps())

    def input_projection_weights(self, layer_type: str) -> int:
        """The weights of the maps that read a layer's input in its attention block. If no weight
        trains below the layer, the backward pass needs no gradient of their input."""
        maps = self.attention_maps(layer_type)
        return _weights({name: maps[name] for name in INPUT_PROJECTIONS[layer_type]})

    @property
    def output_weights(self) -> int:
        """The weights of the output layer: its cost at each position where it runs."""
        return self.vocabulary * self.hidden

    def adapter_weights(self, rank: int) -> int:
        """The weights of GFlowNet's adapters of rank `rank`: for a projection with n inputs and
        m outputs, the two factors [rank, n] and [m, rank]. They are the adapters' cost for each
        token of a forward pass, and the number of GFlowNet's trained parameters."""
        total = 0
        for layer_type in self.layer_types:
            attention = self.attention_maps(layer_type)
            adapted = (
                gfn_policy.FULL_ATTENTION_PROJECTIONS
                if layer_type == FULL_ATTENTION
                else gfn_policy.LINEAR_ATTENTION_PROJECTIONS
            )
            feed_forward = self.feed_forward_maps()
            sizes = [attention[name] for name in adapted]
            sizes += [feed_forward[name] for name in gfn_policy.FEED_FORWARD_PROJECTIONS]
            total += sum(rank * (inputs + outputs) for inputs, outputs in sizes)
        return total

    @property
    def twist_generation_weights(self) -> int:
        """What CTL's twist network costs for one generated token: its two hidden layers and its
        output layer over the whole vocabulary."""
        return 2 * self.hidden**2 + self.vocabulary * self.hidden

    @property
    def twist_update_weights(self) -> int:
        """What an update of the twist network costs for one position of a response, on the
        hidden state that generation kept. Forward: the two hidden layers and the one row of the
        output layer that the response's token picks (2 h^2 + h). Backward: the row's input and
        weights (2 h), the second hidden layer's input and weights (2 h^2), and the first hidden
        layer's weights (h^2); the kept hidden state needs no gradient."""
        return 5 * self.hidden**2 + 3 * self.hidden

    def non_weight_operations(self, layer_type: str, positions: int) -> float:
        """The operations without weights of one layer in a forward pass over `positions`
        positions. Full attention: the scores and the weighted values of each query head over
        the causal context. Linear attention: the decay, the read, the update and the output of
        each head's state at each position, and the convolution."""
        if layer_type == FULL_ATTENTION:
            context = positions * (positions + 1) / 2  # the lengths of the causal contexts, summed
            return 2 * self.attention_heads * self.head_dim * context
        key, value = self._linear_widths
        state = 4 * self.linear_value_heads * self.linear_key_head_dim * self.linear_value_head_dim
        return positions * (state + (2 * key + value) * self.linear_conv_kernel)

    def layer_parameters(self, layer_type: str) -> int:
        """The parameters of one layer: its linear maps, its two norms, and in linear attention
        the convolution, the two vectors of the decay and the norm of the values; in full
        attention the norms of the queries and of the keys."""
        if layer_type == FULL_ATTENTION:
            other = 2 * self.head_dim
        else:
            key, value = self._linear_widths
            other = (
                (2 * key + value) * self.linear_conv_kernel
                + 2 * self.linear_value_heads
                + self.linear_value_head_dim
            )
        return self.layer_weights(layer_type) + other + 2 * self.hidden

    def parameters(self) -> int:
        """The parameters of the model: its layers, the output layer, the embedding if it is a
        matrix of its own, and the last norm."""
        layers = sum(self.layer_parameters(kind) for kind in self.layer_types)
        embedding = 0 if self.tied_embeddings else self.output_weights
        return layers + self.output_weights + embedding + self.hidden

    def last_layers_parameters(self, layers: int) -> int:
        """The parameters of the last `layers` decoder layers, which SFT trains."""
        return sum(self.layer_parameters(kind) for kind in self.last_layers(layers))

    def twist_parameters(self) -> int:
        """The parameters of CTL's twist network: two hidden layers and an output layer over the
        vocabulary, each with a bias."""
        return 2 * (self.hidden**2 + self.hidden) + self.output_weights + self.vocabulary

    def last_layers(self, layers: int) -> tuple[str, ...]:
        """The kinds of the last `layers` decoder layers."""
        if not 1 <= layers <= len(self.layer_types):
            raise ShapeError(f"the model has {len(self.layer_types)} layers, not a last {layers}")
        return self.layer_types[-layers:]


@dataclasses.dataclass(frozen=True)
class Convention:
    """One way to count a pass, where an implementation has a choice.

    prompt: whether the P positions of the prompt count in every pass. They do not if the prompt
        is run once and kept in a cache.
    output_layer_at_every_position: whether the output layer runs at all P + T positions, as a
        teacher-forced pass of transformers does, or only at the T positions whose output is read.
    non_weight_operations: whether the operations without weights count."""

    prompt: bool = True
    output_layer_at_every_position: bool = True
    non_weight_operations: bool = True


CONVENTIONS: dict[str, Convention] = {
    "every_position": Convention(),
    "output_layer_where_read": Convention(output_layer_at_every_position=False),
    "weights_only": Convention(non_weight_operations=False),
    "prompt_cached": Convention(prompt=False, output_layer_at_every_position=False),
}


@dataclasses.dataclass(frozen=True)
class Passes:
    """The cost of each pass over one response, in multiply-accumulates.

    forward: a forward pass of the original model: one inference.
    adapter_forward: a forward pass of GFlowNet's policy, the model with its adapters.
    twisted_generation: generating the response with CTL's twisted model: the model, and the
        twist network at each generated token.
    all_weights_backward: the backward pass when all of the model's weights train.
    adapter_backward: the backward pass of GFlowNet's policy.
    last_layers_backward: the backward pass when only the last layers train.
    twist_update: CTL's update of the twist network on the response, forward and backward."""

    forward: float
    adapter_forward: float
    twisted_generation: float
    all_weights_backward: float
    adapter_backward: float
    last_layers_backward: float
    twist_update: float


def passes(
    shape: Shape,
    *,
    prompt_tokens: int,
    response_tokens: int,
    convention: Convention = Convention(),
    adapter_rank: int = 64,
    trained_last_layers: int = 4,
) -> Passes:
    """The cost of each pass over one response of `response_tokens` tokens after a prompt of
    `prompt_tokens` tokens, counted as `convention` says.

    adapter_rank: the rank of GFlowNet's adapters.
    trained_last_layers: the number of layers that SFT trains."""
    positions = (prompt_tokens if convention.prompt else 0) + response_tokens
    output_positions = positions if convention.output_layer_at_every_position else response_tokens

    def non_weight(layer_types: Sequence[str]) -> float:
        if not convention.non_weight_operations:
            return 0.0
        return sum(shape.non_weight_operations(kind, positions) for kind in layer_types)

    layers = positions * sum(shape.layer_weights(kind) for kind in shape.layer_types)
    output = output_positions * shape.output_weights
    forward = layers + output + non_weight(shape.layer_types)
    adapters = positions * shape.adapter_weights(adapter_rank)
    trained = shape.last_layers(trained_last_layers)
    return Passes(
        forward=forward,
        adapter_forward=forward + adapters,
        twisted_generation=forward + response_tokens * shape.twist_generation_weights,
        # Every map passes the gradient to its input and takes the gradient of its weights.
        all_weights_backward=2 * (layers + output) + 2 * non_weight(shape.layer_types),
        # The frozen maps pass the gradient to their input, except those that read the input of
        # the first layer; each adapter passes it on and takes the gradient of its weights.
        adapter_backward=(
            layers
            - positions * shape.input_projection_weights(shape.layer_types[0])
            + output
            + 2 * adapters
            + 2 * non_weight(shape.layer_types)
        ),
        # The frozen output layer passes the gradient to its input; the trained layers do both,
        # except that the first of them needs no gradient of its input.
        last_layers_backward=(
            output
            + 2 * positions * sum(shape.layer_weights(kind) for kind in trained)
            - positions * shape.input_projection_weights(trained[0])
            + 2 * non_weight(trained)
        ),
        twist_update=response_tokens * shape.twist_update_weights,
    )


@dataclasses.dataclass(frozen=True)
class Factors:
    """What one response costs a method, in inferences (forward passes of the original model
    over one response).

    sampler_draw: generating the response with the method's sampler.
    training_forward: the forward pass of a gradient step on it.
    gradient_response: the rest of a gradient step on it, which the inference count leaves out:
        the backward pass, and for CTL the whole update of the twist network."""

    sampler_draw: float
    training_forward: float
    gradient_response: float


def factors(trains: Trains, cost: Passes) -> Factors:
    """The factors of a method that trains `trains`, from the cost of each pass."""
    one = cost.forward
    if trains == "nothing":
        return Factors(1.0, 1.0, 0.0)
    if trains == "last_layers":
        return Factors(1.0, 1.0, cost.last_layers_backward / one)
    if trains == "adapters":
        adapter_forward = cost.adapter_forward / one
        return Factors(adapter_forward, adapter_forward, cost.adapter_backward / one)
    if trains == "twist":
        # The twist network has no forward pass of its own in an update: `twist_update` is both.
        return Factors(cost.twisted_generation / one, 0.0, cost.twist_update / one)
    return Factors(1.0, 1.0, cost.all_weights_backward / one)
