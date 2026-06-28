# coding=utf-8
# Copyright (C) 2025 THL A29 Limited, a Tencent company and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch HunYuanVL model."""

from typing import Callable, Optional, Tuple, Union, List, Dict

import torch
import torch.utils.checkpoint
from torch import nn


from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs

from ..hunyuan_v1_dense.configuration_hunyuan_v1_dense import HunYuanDenseV1Config
from ..hunyuan_v1_dense.modeling_hunyuan_v1_dense import (
    HunYuanDenseV1Attention,
    HunYuanDenseV1DecoderLayer,
    HunYuanDenseV1MLP,
    HunYuanDenseV1Model,
    HunYuanDenseV1PreTrainedModel,
    HunYuanDenseV1RMSNorm,
    HunYuanDenseV1RotaryEmbedding,
    HunYuanDenseV1ForCausalLM
)

from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    rotate_half,
    repeat_kv,
    eager_attention_forward
)


import json
import types
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from contextlib import contextmanager
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling

logger = logging.get_logger(__name__)


class HunYuanVLVisionConfig(PretrainedConfig):
    model_type = "hunyuan_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_act='gelu',
        hidden_size=1152,
        intermediate_size=4304,
        interpolate_mode='bilinear',
        rms_norm_eps=1e-05,
        learnable_mlp_pooling_size=0,
        num_attention_heads=16,
        num_key_value_heads=None,
        num_channels=3,
        num_hidden_layers=27,
        out_hidden_size=4096,
        patch_size=16,
        remove_prenorm=True,
        spatial_merge_size=2,
        temporal_patch_size=1,
        resize_resolution=2048,
        img_max_token_num=4096,
        max_image_size=2048,
        video_max_image_size=768,
        video_min_image_size=256,
        min_image_size=512,
        anyres_vit_max_image_size=2048,
        max_vit_seq_len=16384,
        text_hidden_size=3072,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.interpolate_mode = interpolate_mode
        self.learnable_mlp_pooling_size = learnable_mlp_pooling_size
        self.num_attention_heads = num_attention_heads
        if not num_key_value_heads:
            self.num_key_value_heads = num_attention_heads
        else:
            self.num_key_value_heads = num_key_value_heads
        self.num_channels = num_channels
        self.num_hidden_layers = num_hidden_layers
        self.out_hidden_size = out_hidden_size
        self.patch_size = patch_size
        self.remove_prenorm = remove_prenorm
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.rms_norm_eps = rms_norm_eps

        self.resize_resolution = resize_resolution
        self.img_max_token_num = img_max_token_num
        self.max_image_size = max_image_size
        self.min_image_size = min_image_size
        self.video_max_image_size = video_max_image_size
        self.video_min_image_size = video_min_image_size
        self.anyres_vit_max_image_size = anyres_vit_max_image_size
        self.max_vit_seq_len = max_vit_seq_len
        self.text_hidden_size = text_hidden_size


class HunYuanVLTextConfig(HunYuanDenseV1Config):
    model_type = "hunyuan_vl_text"
    keys_to_ignore_at_inference = ["past_key_values"]


class HunYuanVLConfig(PretrainedConfig):
    model_type = "hunyuan_vl"
    sub_configs = {"vision_config": HunYuanVLVisionConfig, "text_config": HunYuanVLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        im_start_id=120118,
        im_end_id=120119,
        image_token_id=120120,
        im_newline_id=120121,
        video_start_id=120122,
        video_end_id=120123,
        **kwargs,
    ):
        # We need to init super() here so that it does not reset values
        # that are in text config to the BaseClass defaults. The Base
        # config has many text related defaults and not all defaults are same as for `HunYuanVLTextConfig`
        super().__init__(**kwargs)

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # For BC use all kwargs to init `TextConfig`
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.im_start_id = im_start_id
        self.im_end_id = im_end_id
        self.im_newline_id = im_newline_id
        self.video_start_id = video_start_id
        self.video_end_id = video_end_id

        self.vision_config.text_hidden_size = self.text_config.hidden_size

        # Attention implementation to use. It sets it recursively on sub-configs so we call it again in the end
        self._attn_implementation = kwargs.pop("attn_implementation", None)

    def __setattr__(self, key, value):
        if (
            (text_config := super().__getattribute__("__dict__").get("text_config")) is not None
            and key not in ["dtype", "_attn_implementation_internal"]
            and key in text_config.__dict__
        ):
            setattr(text_config, key, value)
        else:
            super().__setattr__(key, value)

    def __getattribute__(self, key):
        if "text_config" in super().__getattribute__("__dict__") and key not in [
            "_name_or_path",
            "model_type",
            "dtype",
            "_attn_implementation_internal",
        ]:
            text_config = super().__getattribute__("text_config")
            if key in text_config.__dict__:
                return getattr(text_config, key)

        return super().__getattribute__(key)


class HunYuanVisionMLP(nn.Module):
    def __init__(self, config: HunYuanVLConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.dense_h_to_4h = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.dense_4h_to_h = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)

    def forward(self, x):
        intermediate = self.dense_h_to_4h(x)
        intermediate = self.act_fn(intermediate)
        output = self.dense_4h_to_h(intermediate)
        return output


class HunYuanVLRMSNorm(LlamaRMSNorm):
    pass

class HunYuanVLMLP(HunYuanDenseV1MLP):
    pass

class HunYuanVisionPatchEmbed(nn.Module):
    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.spatial_merge_size = config.spatial_merge_size
        self.interpolate_mode = config.interpolate_mode

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        self.max_num_patches = (config.max_image_size // self.patch_size) ** 2
        self.num_positions = self.max_num_patches + 1
        self.position_edge = int(self.num_positions ** 0.5)
        # first token is cls token, skip it
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        self.patch_pos_embed = None

    def forward(self, pixel_values: torch.Tensor, grid_thw: list[list[int]]) -> torch.Tensor:
        num_patches, hidden_size = pixel_values.shape
        pixel_values = pixel_values.reshape(num_patches, self.num_channels, self.patch_size, self.patch_size)

        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.squeeze(-1).squeeze(-1).unsqueeze(0)

        if self.patch_pos_embed is None:
            patch_pos_shape = (1, self.position_edge, self.position_edge, self.embed_dim)
            self.patch_pos_embed = (
                self.position_embedding.weight[1:, :].reshape(patch_pos_shape).permute(0, 3, 1, 2).float()
            )

        patch_pos_embed_list = []
        for grid in grid_thw:
            _, h0, w0 = grid
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            h0, w0 = h0 + 0.1, w0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                self.patch_pos_embed,
                scale_factor=((h0 / self.position_edge).item(), (w0 / self.position_edge).item()),
                mode=self.interpolate_mode,
                align_corners=False,
            )

            patch_pos_embed = (
                patch_pos_embed.reshape(self.embed_dim, -1).transpose(0, 1).unsqueeze(0).to(patch_embeds.dtype)
            )
            patch_pos_embed_list.append(patch_pos_embed)

        patch_pos_embed = torch.cat(patch_pos_embed_list, dim=1)
        embeddings = patch_embeds + patch_pos_embed

        return embeddings


class HunYuanVisionPatchMerger(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_merge_size,
        rms_norm_eps,
        **kwargs,
    ):
        super().__init__()

        embed_std = out_channels ** -0.5
        self.spatial_merge_size = spatial_merge_size
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=spatial_merge_size, stride=spatial_merge_size),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=1),
        )
        self.mlp = nn.Linear(in_channels * 4, out_channels)
        self.image_newline = nn.Parameter(torch.randn(in_channels * 4) * embed_std)
        self.image_begin = nn.Parameter(torch.randn(out_channels) * embed_std)
        self.image_end = nn.Parameter(torch.randn(out_channels) * embed_std)
        self.image_sep = nn.Parameter(torch.randn(out_channels) * embed_std)

        self.before_rms = HunYuanVLRMSNorm(in_channels, eps=rms_norm_eps)
        self.after_rms = HunYuanVLRMSNorm(out_channels, eps=rms_norm_eps)

    def forward(self, x, size=(16, 16)):
        x = self.before_rms(x)
        h, w = size
        dtype = x.dtype
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, int(h.item()), int(w.item()))
        x = self.proj(x)  # b,c,h,w
        b, c, h, w = x.shape
        x = torch.cat(
            [x, self.image_newline.reshape(1, c, 1, 1).expand(b, c, h, 1).to(dtype, non_blocking=True)], dim=-1
        )
        x = x.reshape(b, c, -1).permute(0, 2, 1)
        x = self.mlp(x)

        begin = self.image_begin.reshape(1, 1, -1).expand(b, 1, x.shape[-1]).to(dtype, non_blocking=True)
        end = self.image_end.reshape(1, 1, -1).expand(b, 1, x.shape[-1]).to(dtype, non_blocking=True)
        x = torch.cat([begin, x, end], dim=1)

        return self.after_rms(x)


class HunYuanVisionAttention(nn.Module):
    def __init__(self, config: HunYuanVLConfig):
        super().__init__()
        self.config = config
        self.is_causal = False   # used in flash_attention
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=True
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class HunYuanVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = HunYuanVisionAttention(config)
        self.mlp = HunYuanVisionMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class HunYuanVisionTransformer(nn.Module):
    config: HunYuanVLVisionConfig
    _no_split_modules = ["HunYuanVLVisionBlock"]

    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = HunYuanVisionPatchEmbed(config)
        self.layers = nn.ModuleList(
            [HunYuanVisionBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.perceive = HunYuanVisionPatchMerger(
            self.config.hidden_size,
            self.config.text_hidden_size,
            self.config.spatial_merge_size,
            self.config.rms_norm_eps,
        )

    def get_activation_function(self, act_name: str):
        act_map = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }
        return act_map.get(act_name.lower(), nn.GELU())  # default GELU

    # @auto_docstring
    def forward(
        self,
        x: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        #
        r"""
        grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width dimensions of feature shape for each image. Each row contains [t, h, w] values.
        """
        hidden_states = self.embeddings(x, grid_thw)
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        cu_seqlens: list = [0]
        for t, h, w in grid_thw:
            cu_seqlens.append((h * w).item())

        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        cu_seqlens = torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32)
        split_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        split_items = torch.split(hidden_states, split_lengths, dim=1)

        processed_items = []
        for grid, item in zip(grid_thw, split_items):
            t, h, w = grid
            processed = self.perceive(item, size=(h, w))
            processed_items.append(processed)

        hidden_states = torch.cat(processed_items, dim=1)

        return hidden_states


def apply_rotary_pos_emb_xdrope(q, k, cos, sin, position_ids, xdrope_section, output_size=None):
    """Applies XD Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`): The position IDs for the tokens.
        xdrope_section (`list`): The section ratios for XD RoPE.
        output_size (`tuple`, optional): The output size of the tensors. Defaults to None.
        bf16 (bool, optional): Whether to use bfloat16 precision. Defaults to False.

    Returns:
        `tuple(torch.Tensor)`: The query and key tensors rotated using the XD Rotary Position Embedding.
    """
    x_dim = len(xdrope_section)
    cos = cos[position_ids, ...].permute(0, 2, 1, 3).reshape(output_size[0], output_size[2], x_dim, -1).contiguous()
    sin = sin[position_ids, ...].permute(0, 2, 1, 3).reshape(output_size[0], output_size[2], x_dim, -1).contiguous()

    xdrope_section = xdrope_section * 2

    # for xd concat
    assert sum(xdrope_section) == cos.shape[-1], "Illegal partition for xd rope"
    cos = torch.cat([m[:, :, i % x_dim, :] for i, m in enumerate(cos.split(xdrope_section, dim=-1))], dim=-1)
    sin = torch.cat([m[:, :, i % x_dim, :] for i, m in enumerate(sin.split(xdrope_section, dim=-1))], dim=-1)

    # for head repeat
    cos = cos.view(output_size[0], 1, output_size[2], -1)  # .repeat(1, output_size[1], 1, 1)
    sin = sin.view(output_size[0], 1, output_size[2], -1)  # .repeat(1, output_size[1], 1, 1)

    origin_dtype = q.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.float(), sin.float()
    q_out, k_out = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

    return q_out.to(origin_dtype), k_out.to(origin_dtype)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: Optional[torch.Tensor]=None, unsqueeze_dim: int=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    else:
        cos = cos.unsqueeze(0).unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(0).unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class HunYuanVLRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: HunYuanVLConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type if self.rope_type != "xdrope" else "dynamic"]
        if self.rope_type in ["xdrope", "dynamic"] and config.rope_scaling["alpha"]:
            # DynamicNTKAlphaRotary
            self.dim = config.head_dim
            base = config.rope_theta * config.rope_scaling.get("alpha") ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.attention_scaling = 1.0
        else:
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        self._set_cos_sin_cache(
            seq_len=config.max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1).float()
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len: Optional[int]=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class HunYuanVLAttention(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_causal = True  # used in flash_attention
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.query_layernorm = HunYuanVLRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_layernorm = HunYuanVLRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = HunYuanVLRotaryEmbedding(config=config)
        self.xdrope_section = config.rope_scaling['xdrope_section']

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        origin_kv_seq_len = key_states.shape[-2]
        if past_key_values is not None:
            kv_seq_len += past_key_values.get_seq_length(self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        if self.xdrope_section is not None:
            if past_key_values is None or past_key_values.get_seq_length() == 0:
                output_size = (
                    query_states.size(0),
                    query_states.size(1),
                    query_states.size(2),
                    key_states.size(2),
                )
                query_states, key_states = apply_rotary_pos_emb_xdrope(
                    query_states, key_states, cos, sin, position_ids, self.xdrope_section, output_size
                )
            else:
                position_ids = (
                    torch.ones(position_ids.shape[0], 1, dtype=torch.long, device=position_ids.device)
                    * past_key_values.get_seq_length()
                )
                cos, sin = cos[-origin_kv_seq_len:, :], sin[-origin_kv_seq_len:, :]
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            position_ids = torch.ones(
                position_ids.shape[0], 1, dtype=torch.long, device=position_ids.device
            ) * past_key_values.get_seq_length(self.layer_idx)
            cos, sin = cos[-origin_kv_seq_len:, :], sin[-origin_kv_seq_len:, :]
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class HunYuanVLDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config: Union[HunYuanVLVisionConfig, HunYuanVLTextConfig],
        layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        if config.norm_type == 'hf_rms' or config.norm_type == 'rms':
            self.input_layernorm = HunYuanVLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = HunYuanVLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        elif config.norm_type == 'fused' or config.norm_type == 'torch_nn':
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            assert False, "other norm_type are not supported"


class HunYuanVLPreTrainedModel(LlamaPreTrainedModel):
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


@auto_docstring
class HunYuanVLModel(HunYuanVLPreTrainedModel):
    def __init__(self, config: Union[HunYuanVLConfig, HunYuanVLTextConfig]):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [HunYuanVLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = HunYuanVLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    @check_model_inputs
    # @auto_docstring # TODO Fix this
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        hidden_states = inputs_embeds
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

class HunYuanVLForCausalLM(LlamaForCausalLM):
    pass

class HunYuanVLForConditionalGeneration(HunYuanVLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    config: HunYuanVLConfig

    def __init__(self, config: HunYuanVLConfig):
        super().__init__(config)
        self.model = HunYuanVLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vit = HunYuanVisionTransformer(config.vision_config)
        self.config = config
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
        >>> from PIL import Image
        >>> import torch

        >>> model_name_or_path = "tencent/HunyuanOCR"
        >>> processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=False)
        >>> model = HunYuanVLForConditionalGeneration.from_pretrained(
        ...     model_name_or_path,
        ...     attn_implementation="eager",
        ...     torch_dtype=torch.bfloat16,
        ...     device_map="auto",
        ... )

        >>> img_path = "path/to/your/image.jpg"
        >>> image = Image.open(img_path).convert("RGB")

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "image": img_path},
        ...             {"type": "text", "text": "Extract the text from the image."},
        ...         ],
        ...     }
        ... ]
        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

        >>> with torch.no_grad():
        ...     generated_ids = model.generate(**inputs, max_new_tokens=1024)
        >>> generated_ids_trimmed = generated_ids[0][len(inputs["input_ids"][0]):]
        >>> output = processor.decode(generated_ids_trimmed, skip_special_tokens=True)

        >>> print(output)

        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # def prepare_inputs_for_generation(
    #     self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    # ):
    #     inputs = super().prepare_inputs_for_generation(
    #         input_ids,
    #         past_key_values=past_key_values,
    #         attention_mask=attention_mask,
    #         inputs_embeds=inputs_embeds,
    #         **kwargs,
    #     )
    #     return inputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        imgs: Optional[list[torch.FloatTensor]] = None,
        imgs_pos: Optional[list[int]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[list[int]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        inputs_embeds = self.model.embed_tokens(input_ids)

        if self.vit is not None and pixel_values is not None:
            pixel_values = pixel_values.to(torch.bfloat16)
            image_embeds = self.vit(pixel_values, image_grid_thw)

            # ViT may be deployed on different GPUs from those used by LLMs, due to auto-mapping of accelerate.
            image_embeds = image_embeds.to(input_ids.device, non_blocking=True)

            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        return super().generate(
            inputs=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            # eos_token_id=self.config.eod_token_id,
            **kwargs,
        )

    # Copied from transformers.models.llava.modeling_llava.LlavaModel.get_placeholder_mask
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        return special_image_mask, None


__all__ = [
    "HunYuanVLConfig",
    "HunYuanVLVisionConfig",
    "HunYuanVLTextConfig",
    "HunYuanVLForConditionalGeneration",
    "HunYuanVLForCausalLM",
    "HunYuanVLModel",
    "HunYuanVLPreTrainedModel",
    "HunYuanVLTextModel"
]
