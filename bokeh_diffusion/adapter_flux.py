import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FluxTransformer2DModel
from huggingface_hub import PyTorchModelHubMixin
from diffusers.models.activations import get_activation


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        network_alpha: float = None,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class FluxAttnProcessor2_0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, bokeh_embeds=None, attention_mask=None, image_rotary_emb=None):
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class BokehFluxAttnProcessor2_0(torch.nn.Module):
    def __init__(
        self,
        context_dim,
        hidden_dim,
        block_name=None,
        bokeh_scale=1.0,
        unfreeze_q=False,
        unfreeze_k=False,
        lora_rank=None,
        lora_scale=1.0,
        lora_alpha=None
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("BokehFluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.block_name = block_name
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.bokeh_scale = bokeh_scale
        self.lora_scale = lora_scale
        self.unfreeze_q = unfreeze_q
        self.unfreeze_k = unfreeze_k

        self.to_k_bokeh = LoRALinear(context_dim, hidden_dim, rank=lora_rank, network_alpha=lora_alpha)
        self.to_v_bokeh = LoRALinear(context_dim, hidden_dim, rank=lora_rank, network_alpha=lora_alpha)
        if self.unfreeze_q:
            self.to_q_adp = LoRALinear(hidden_dim, hidden_dim, rank=lora_rank, network_alpha=lora_alpha)
        if self.unfreeze_k:
            self.to_k_adp = LoRALinear(hidden_dim, hidden_dim, rank=lora_rank, network_alpha=lora_alpha)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        bokeh_embeds=None,
        attention_mask=None,
        image_rotary_emb=None,
        perform_swap=False, 
        batch_swap_ids=None
    ):
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)  # [batch_size, seq_len, hidden_dim]
        key = attn.to_k(hidden_states)  # [batch_size, seq_len, hidden_dim]
        value = attn.to_v(hidden_states)  # [batch_size, seq_len, hidden_dim]

        if self.unfreeze_q:
            query = query + self.lora_scale * self.to_q_adp(query)
        if self.unfreeze_k:
            key = key + self.lora_scale * self.to_k_adp(key)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        
        if perform_swap:
            query = query[batch_swap_ids]

        if perform_swap and self.unfreeze_k:
            key_swaps  = key[batch_swap_ids]
            key  = torch.cat([key, key_swaps], dim=2)
            value = torch.cat([value, value], dim=2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if bokeh_embeds is not None:
            key_dof = self.to_k_bokeh(bokeh_embeds)
            value_dof = self.to_v_bokeh(bokeh_embeds)
            key_dof = key_dof.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value_dof = value_dof.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            camera_hidden_states = F.scaled_dot_product_attention(query, key_dof, value_dof, attn_mask=None, dropout_p=0.0, is_causal=False)
            camera_hidden_states = camera_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            camera_hidden_states = camera_hidden_states.to(query.dtype)
            hidden_states = hidden_states + self.bokeh_scale * camera_hidden_states
        
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class BokehFluxControlAdapter(
    torch.nn.Module,
    PyTorchModelHubMixin,
):
    def __init__(
        self,
        base_model,
        blocks,
        attn_hidden_dim=3072,
        bokeh_input_dim=1,
        bokeh_hidden_dim=64,
        bokeh_context_dim=768,
        lora_rank=128,
        lora_alpha=None,
        lora_scale=1.0,
        bokeh_scale=1.0,
        unfreeze_q=False,
        unfreeze_k=False,
    ):
        super().__init__()
        self.blocks = blocks
        self.bokeh_scale = bokeh_scale
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scale = lora_scale
        self.attn_hidden_dim = attn_hidden_dim
        self.bokeh_input_dim = bokeh_input_dim
        self.bokeh_hidden_dim = bokeh_hidden_dim
        self.bokeh_context_dim = bokeh_context_dim
        self.unfreeze_q = unfreeze_q
        self.unfreeze_k = unfreeze_k

        self.embedding_layer = nn.Sequential(
            nn.Linear(self.bokeh_input_dim, self.bokeh_hidden_dim),
            get_activation("relu"),
            nn.Linear(self.bokeh_hidden_dim, self.bokeh_context_dim)
        )
                
        self._attach_transformer(base_model, blocks)

    def _attach_transformer(self, base_model, blocks):
        attn_procs = {}
        for name in base_model.attn_processors.keys():
            block_name = name.split(".attn")[0]
            if block_name in blocks:
                attn_procs[name] = BokehFluxAttnProcessor2_0(
                    context_dim=self.bokeh_context_dim,
                    hidden_dim=self.attn_hidden_dim,
                    block_name=block_name,
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    unfreeze_q=self.unfreeze_q,
                    unfreeze_k=self.unfreeze_k,
                )
            else:
                attn_procs[name] = FluxAttnProcessor2_0()
        
        base_model.set_attn_processor(attn_procs)
        self.adapter_modules = torch.nn.ModuleList(base_model.attn_processors.values())

    def set_lora_scale(self, lora_scale):
        self.lora_scale = lora_scale
        for module in self.adapter_modules:
            module.lora_scale = lora_scale

    def set_bokeh_scale(self, bokeh_scale):
        self.bokeh_scale = bokeh_scale
        for module in self.adapter_modules:
            module.bokeh_scale = bokeh_scale

    def forward(
        self,
        base_model: FluxTransformer2DModel,
        bokeh_ann: torch.Tensor,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor,
        pooled_projections: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor,
        perform_swap: bool = False,
        batch_swap_ids: list[int] = None
    ):
        bokeh_embeds = self.embedding_layer(bokeh_ann)
        noise_pred = base_model(
            hidden_states,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_projections,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids,
            joint_attention_kwargs={
                "bokeh_embeds": bokeh_embeds,
                "perform_swap": perform_swap,
                "batch_swap_ids": batch_swap_ids,
            },
            return_dict=False,
        )[0]
        return noise_pred
