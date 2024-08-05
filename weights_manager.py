import json
import os
from diffusers.models.attention_processor import LoRAAttnProcessor2_0, AttnProcessor2_0, LoRALinearLayer
from peft import get_peft_model, LoraConfig
from safetensors.torch import load_file
from dataset_and_utils import TokenEmbeddingsHandler
from weights import WeightsDownloadCache


class WeightsManager:
    def __init__(self, predictor):
        self.predictor = predictor
        self.weights_cache = WeightsDownloadCache()

    def load_trained_weights(self, weights, pipe):
        from no_init import no_init_or_tensor

        # weights can be a URLPath, which behaves in unexpected ways
        weights = str(weights)
        if self.predictor.tuned_weights == weights:
            print("skipping loading .. weights already loaded")
            return

        self.predictor.tuned_weights = weights

        local_weights_cache = self.weights_cache.ensure(weights)

        # load UNET
        print("Loading fine-tuned model")
        self.predictor.is_lora = False

        maybe_unet_path = os.path.join(local_weights_cache, "unet.safetensors")
        if not os.path.exists(maybe_unet_path):
            print("Does not have Unet. assume we are using LoRA")
            self.predictor.is_lora = True

        if not self.predictor.is_lora:
            print("Loading Unet")

            new_unet_params = load_file(
                os.path.join(local_weights_cache, "unet.safetensors")
            )
            # this should return _IncompatibleKeys(missing_keys=[...], unexpected_keys=[])
            pipe.unet.load_state_dict(new_unet_params, strict=False)

        else:
            print("Loading Unet LoRA")

            unet = pipe.unet

            # tensors = load_file(os.path.join(local_weights_cache, "lora.safetensors"))
            #
            # unet_lora_attn_procs = {}
            # name_rank_map = {}
            # for tk, tv in tensors.items():
            #     # up is N, d
            #     if tk.endswith("up.weight"):
            #         proc_name = ".".join(tk.split(".")[:-3])
            #         r = tv.shape[1]
            #         name_rank_map[proc_name] = r
            #
            # for name, attn_processor in unet.attn_processors.items():
            #     cross_attention_dim = (
            #         None
            #         if name.endswith("attn1.processor")
            #         else unet.config.cross_attention_dim
            #     )
            #     if name.startswith("mid_block"):
            #         hidden_size = unet.config.block_out_channels[-1]
            #     elif name.startswith("up_blocks"):
            #         block_id = int(name[len("up_blocks.")])
            #         hidden_size = list(reversed(unet.config.block_out_channels))[
            #             block_id
            #         ]
            #     elif name.startswith("down_blocks"):
            #         block_id = int(name[len("down_blocks.")])
            #         hidden_size = unet.config.block_out_channels[block_id]
            #     with no_init_or_tensor():
            #         module = LoRAAttnProcessor2_0(
            #             hidden_size=hidden_size,
            #             cross_attention_dim=cross_attention_dim,
            #             rank=name_rank_map[name],
            #         )
            #
            #         # module = AttnProcessor2_0()
            #         # def create_lora_layer(dim, rank):
            #         #     return LoRALinearLayer(dim, dim, rank)
            #         #
            #         # # Then, set the LoRA layers directly on the attention components
            #         # module.to_q.lora_layer = create_lora_layer(hidden_size, rank=name_rank_map[name])
            #         # module.to_k.lora_layer = create_lora_layer(hidden_size, rank=name_rank_map[name])
            #         # module.to_v.lora_layer = create_lora_layer(hidden_size, rank=name_rank_map[name])
            #         # module.to_out[0].lora_layer = create_lora_layer(hidden_size, rank=name_rank_map[name])
            #         #
            #         # # If you're using cross-attention, you might also need to set:
            #         # if cross_attention_dim is not None:
            #         #     module.to_k_cross.lora_layer = create_lora_layer(cross_attention_dim, rank=name_rank_map[name])
            #         #     module.to_v_cross.lora_layer = create_lora_layer(cross_attention_dim, rank=name_rank_map[name])
            #
            #     unet_lora_attn_procs[name] = module.to("cuda", non_blocking=True)
            #
            # unet.set_attn_processor(unet_lora_attn_procs)
            # unet.load_state_dict(tensors, strict=False)
            lora_state_dict = load_file(os.path.join(local_weights_cache, "lora.safetensors"))

            # Determine the rank from the loaded weights
            rank = None
            for key, value in lora_state_dict.items():
                if key.endswith("up.weight"):
                    rank = value.shape[1]
                    break

            if rank is None:
                raise ValueError("Could not determine LoRA rank from weights")

            # Define LoRA config
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=rank,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                lora_dropout=0.0,
                bias="none",
            )

            # Apply LoRA to the model
            unet = get_peft_model(unet, lora_config)

            # Load the LoRA weights
            unet.load_state_dict(lora_state_dict, strict=False)

            # Set the UNet back to the pipeline
            pipe.unet = unet

        # load text
        handler = TokenEmbeddingsHandler(
            [pipe.text_encoder, pipe.text_encoder_2], [pipe.tokenizer, pipe.tokenizer_2]
        )
        handler.load_embeddings(os.path.join(local_weights_cache, "embeddings.pti"))

        # load params
        with open(os.path.join(local_weights_cache, "special_params.json"), "r") as f:
            params = json.load(f)
        self.predictor.token_map = params

        self.predictor.tuned_model = True
