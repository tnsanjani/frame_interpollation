import abc
import torch
from typing import Tuple, List
from einops import rearrange

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        # consistent structure for all stores
        return {
            "down_cross": [], "mid_cross": [], "up_cross": [],
            "down_self": [], "mid_self": [], "up_self": []
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # append the raw attention tensor for this step/layer
        self.step_store.setdefault(key, []).append(attn)
        return attn

    def between_steps(self):
        """
        Called when a diffusion timestep completes. Finalize step_store into attention_store and
        optionally accumulate into global_store (for averaging across steps).
        """
        # finalize current step as the attention_store for inspection
        self.attention_store = {k: [t for t in v] for k, v in self.step_store.items()}

        if self.save_global_store:
            # accumulate step_store into global_store safely (detach to avoid graph retention)
            if not any(len(v) for v in self.global_store.values()):
                # copy tensors (detach + clone)
                self.global_store = {k: [t.detach().clone() for t in v] for k, v in self.step_store.items()}
            else:
                for key in self.get_empty_store().keys():
                    step_list = self.step_store.get(key, [])
                    global_list = self.global_store.get(key, [])
                    # ensure global list has at least as many entries as step_list
                    for i, t in enumerate(step_list):
                        if i >= len(global_list):
                            # append detached clone if missing
                            global_list.append(t.detach().clone())
                        else:
                            global_list[i] = global_list[i] + t.detach()
                    # write back
                    self.global_store[key] = global_list

        # reset step store for next timestep
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        # return the last completed step's store (per-step)
        return self.attention_store

    def get_average_global_attention(self):
        # compute average across all saved steps (if any)
        if not any(len(v) for v in self.global_store.values()):
            return self.get_empty_store()
        denom = max(1, self.cur_step)
        averaged = {}
        for key in self.get_empty_store().keys():
            averaged[key] = []
            for item in self.global_store.get(key, []):
                averaged[key].append(item / denom)
        return averaged

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = self.get_empty_store()
        self.global_store = self.get_empty_store()

    def __init__(self, save_global_store=False):
        '''
        Initialize an empty AttentionStore
        '''
        super(AttentionStore, self).__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = self.get_empty_store()
        self.global_store = self.get_empty_store()
        self.curr_step_index = 0


class AttentionStoreProcessor:

    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # store attention in shape (b, h, i, j)
        self.attnstore(rearrange(attention_probs, '(b h) i j -> b h i j', b=batch_size), False, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AttentionFlipCtrlProcessor:

    def __init__(self, attnstore, attnstore_ref, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        # rename to attnstore_ref to avoid confusion/typo
        self.attnstore_ref = attnstore_ref
        self.place_in_unet = place_in_unet

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        ref_store = self.attnstore_ref.attention_store
        key_self = f"{self.place_in_unet}_self"

        if key_self not in ref_store or len(ref_store[key_self]) == 0:
            self.attnstore(rearrange(attention_probs, '(b h) i j -> b h i j', b=batch_size), False, self.place_in_unet)
        else:
            down_len = len(ref_store.get("down_self", []))
            mid_len = len(ref_store.get("mid_self", []))
            up_len = len(ref_store.get("up_self", []))
            total = down_len + mid_len + up_len

            if total == 0:
                self.attnstore(rearrange(attention_probs, '(b h) i j -> b h i j', b=batch_size), False, self.place_in_unet)
            else:
                global_idx = max(0, min(self.attnstore.cur_att_layer, total - 1))
                if self.place_in_unet == "down":
                    if global_idx < down_len:
                        local_idx = global_idx
                    else:
                        local_idx = down_len - 1  
                elif self.place_in_unet == "mid":
                    local_idx = global_idx - down_len
                    if not (0 <= local_idx < mid_len):
    
                        local_idx = max(0, mid_len - 1)
                elif self.place_in_unet == "up":
                    local_idx = global_idx - (down_len + mid_len)
                    if not (0 <= local_idx < up_len):
                        local_idx = max(0, up_len - 1)
                else:
                    local_idx = max(0, len(ref_store[key_self]) - 1)
                local_idx = max(0, min(local_idx, len(ref_store[key_self]) - 1))

                attention_probs_ref = ref_store[key_self][local_idx]
                attention_probs_ref = rearrange(attention_probs_ref, 'b h i j -> (b h) i j')
                attention_probs_ref = attention_probs_ref.to(attention_probs.device, dtype=attention_probs.dtype)
                attention_probs = torch.flip(attention_probs_ref, dims=(-2, -1))

                self.attnstore(rearrange(attention_probs, '(b h) i j -> b h i j', b=batch_size), False, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def register_temporal_self_attention_control(unet, controller):
    attn_procs = {}
    temporal_self_att_count = 0
    for name in unet.attn_processors.keys():
        if name.endswith("temporal_transformer_blocks.0.attn1.processor"):
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                place_in_unet = "down"
            else:
                continue

            temporal_self_att_count += 1
            attn_procs[name] = AttentionStoreProcessor(
                attnstore=controller, place_in_unet=place_in_unet
            )
        else:
            attn_procs[name] = unet.attn_processors[name]

    unet.set_attn_processor(attn_procs)
    controller.num_att_layers = temporal_self_att_count


def register_temporal_self_attention_flip_control(unet, controller, controller_ref):
    attn_procs = {}
    temporal_self_att_count = 0
    for name in unet.attn_processors.keys():
        if name.endswith("temporal_transformer_blocks.0.attn1.processor"):
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                place_in_unet = "down"
            else:
                continue

            temporal_self_att_count += 1
            attn_procs[name] = AttentionFlipCtrlProcessor(
                attnstore=controller, attnstore_ref=controller_ref, place_in_unet=place_in_unet
            )
        else:
            attn_procs[name] = unet.attn_processors[name]

    unet.set_attn_processor(attn_procs)
    controller.num_att_layers = temporal_self_att_count
