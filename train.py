import logging
import math
import os
import shutil
from glob import glob
from pathlib import Path
from PIL import Image
import xformers
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from einops import rearrange
import transformers
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
import copy
from parse_args import parse_args
from data import StereoEventDataset

import diffusers
from diffusers import AutoencoderKLTemporalDecoder
from diffusers import  UNetSpatioTemporalConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing


from custom_diffusers.pipelines.pipeline_stable_video_diffusion_with_ref_attnmap import StableVideoDiffusionWithRefAttnMapPipeline
from custom_diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from attn_ctrl.attention_control import (AttentionStore, register_temporal_self_attention_control, register_temporal_self_attention_flip_control)
from attn_ctrl.lefusion2 import LatentEventFusion

logger = get_logger(__name__, log_level="INFO")
model_name =  "stabilityai/stable-video-diffusion-img2vid-xt"

def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()

n=3
weight_dtype = torch.float16 
n_frames = 15
noise_aug_strength = 0.02
fps=7   

def main():
    args = parse_args()
    
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,mixed_precision=args.mixed_precision,log_with=args.report_to,project_config=accelerator_project_config)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()

    set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = EulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(model_name, subfolder="feature_extractor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_name, subfolder="image_encoder", variant=args.variant)
    vae = AutoencoderKLTemporalDecoder.from_pretrained(model_name, subfolder="vae", variant=args.variant)
    unet = UNetSpatioTemporalConditionModel.from_pretrained(model_name, subfolder="unet", low_cpu_mem_usage=True, variant=args.variant)
    ref_unet = copy.deepcopy(unet)
    lef_model = LatentEventFusion(in_channels=4, out_channels=4)
    vae.to(dtype=torch.float16)

    lef_model = lef_model.to(accelerator.device).to(dtype=weight_dtype)

    lef_model.to(unet.device)
    lef_model.requires_grad_(True)

    controller_ref = AttentionStore()
    register_temporal_self_attention_control(ref_unet, controller_ref)

    controller = AttentionStore()
    register_temporal_self_attention_flip_control(unet, controller, controller_ref)
    ref_unet.requires_grad_(False)
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)

    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    ref_unet.to(accelerator.device, dtype=weight_dtype)
    lef_model.to(accelerator.device, weight_dtype)

    unet_train_params_list = []
    for name, para in unet.named_parameters():
        if 'temporal_transformer_blocks.0.attn1.to_v.weight' in name or 'temporal_transformer_blocks.0.attn1.to_out.0.weight' in name:
            unet_train_params_list.append(para)
            para.requires_grad = True
        else:
            para.requires_grad = False
    
    if args.mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)
 
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            model = models.pop()
            load_model = UNetSpatioTemporalConditionModel.from_pretrained(input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
            del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)
    optimizer = torch.optim.AdamW( unet_train_params_list,lr=args.learning_rate,betas=(args.adam_beta1, args.adam_beta2),weight_decay=args.adam_weight_decay,eps=args.adam_epsilon)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    train_dataset = StereoEventDataset(video_data_dir="/home/nthadishetty1/frame_interpollation/depth_event_rgd_data",frame_height=375,frame_width=375)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,shuffle=None,collate_fn=None,batch_size=1, num_workers=1)


    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(args.lr_scheduler,optimizer=optimizer,num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,num_training_steps=max_train_steps * accelerator.num_processes)
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)

    if accelerator.is_main_process:
        accelerator.init_trackers("image2video-reverse-fine-tune", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(range(0, max_train_steps),initial=initial_global_step,desc="Steps",disable=not accelerator.is_local_main_process)

    def _get_add_time_ids(
        dtype,
        batch_size,
        fps=6,
        motion_bucket_id=127,
        noise_aug_strength=0.02,  
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
        passed_add_embed_dim = unet.module.config.addition_time_embed_dim * \
            len(add_time_ids)
        expected_add_embed_dim = unet.module.add_embedding.linear_1.in_features
        assert (expected_add_embed_dim == passed_add_embed_dim)

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids

    def compute_image_embeddings(image):
        image = _resize_with_antialiasing(image, (224, 224))
        image = (image + 1.0) / 2.0
        image = feature_extractor(images=image,do_normalize=True,do_center_crop=False,do_resize=False,do_rescale=False,return_tensors="pt").pixel_values
        image = image.to(accelerator.device).to(dtype=weight_dtype)
        image_embeddings = image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)
        return image_embeddings
      
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx >= n:
                break
            video_name = batch['video_name'][0]     
            left_data = batch['left']
            right_data = batch['right']
            first_left_rgb = left_data['pixel_values'][:, 0].to(accelerator.device).to(dtype=weight_dtype)
            last_left_rgb = left_data['pixel_values'][:, -1].to(accelerator.device).to(dtype=weight_dtype)
            right_events = right_data['events'].to(accelerator.device).to(dtype=weight_dtype)
            #print(f' first and last rgb {first_left_rgb.shape, last_left_rgb.shape}')

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                encoder_hidden_states_ref = compute_image_embeddings(first_left_rgb) 
                encoder_hidden_states = compute_image_embeddings(last_left_rgb)
                
                left_data['pixel_values'] = left_data['pixel_values'].to(accelerator.device).to(dtype=weight_dtype)
                right_events = right_events.to(accelerator.device).to(dtype=weight_dtype)
                evs_latents = right_events[:, :15]
                evs_latents =  evs_latents.to(accelerator.device)
                noise =  torch.randn_like(last_left_rgb)

                first_left_rgb_latent = first_left_rgb 
                conditions_latent = vae.encode(first_left_rgb_latent).latent_dist.mode()
                conditions_latent = conditions_latent.unsqueeze(1).repeat(1, args.num_frames, 1, 1, 1)

                conditions_ref = last_left_rgb
                conditions_latent_ref = vae.encode(conditions_ref).latent_dist.mode()
                conditions_latent_ref = conditions_latent_ref.unsqueeze(1).repeat(1, args.num_frames, 1, 1, 1)
                print(f'last_left_rgb_latent shape, after latent conversion is {conditions_latent_ref.shape}, first_left_rgb_latent shape, after latent conversion is {conditions_latent.shape}')

                left_data['pixel_values'] = left_data['pixel_values'][:,:15]
                pixel_values = rearrange(left_data['pixel_values'], "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                #print(latents.shape)
                latents = rearrange(latents, "(b f) c h w -> b f c h w", f=args.num_frames)
                print(f' latents shape is {latents.shape}')

                noise = torch.randn_like(latents)
                if args.noise_offset:
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], latents.shape[2], 1, 1), device=latents.device)

                bsz = latents.shape[0]
                sigmas = rand_log_normal(shape=[bsz,], loc=0.7, scale=1.6).to(latents.device)
                sigmas = sigmas[:, None, None, None, None]
                timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(accelerator.device)
                
                noisy_latents = latents + noise * sigmas
                noisy_latents_inp = noisy_latents / ((sigmas**2 + 1) ** 0.5)
                new_condition = lef_model(conditions_latent, evs_latents, conditions_latent_ref)
                new_condition = new_condition.to(weight_dtype) 
                cat_condition = new_condition.clone().detach()

                noisy_latents_inp = torch.cat([noisy_latents_inp, cat_condition], dim=2)
                target = latents
                    
                added_time_ids = _get_add_time_ids(encoder_hidden_states.dtype, bsz).to(accelerator.device)
                encoder_hidden_states_fused = (encoder_hidden_states + encoder_hidden_states_ref)/2.0
                model_pred = unet(noisy_latents_inp.to(weight_dtype), timesteps.to(weight_dtype),encoder_hidden_states=encoder_hidden_states_fused,added_time_ids=added_time_ids,return_dict=False)[0]
                c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                c_skip = 1 / (sigmas**2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                weighing = (1 + sigmas ** 2) * (sigmas**-2.0)
                loss = torch.mean((weighing.float() * (denoised_latents.float() -target.float()) ** 2).reshape(target.shape[0], -1),dim=1,)
                loss = loss.mean()
                
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet_train_params_list
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if global_step >= max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float16)

        unwrapped_unet = unwrap_model(unet)
        pipeline = StableVideoDiffusionWithRefAttnMapPipeline.from_pretrained(model_name,scheduler=noise_scheduler,unet=unwrapped_unet,variant=args.variant)
        pipeline.save_pretrained(args.output_dir)    
        lef_model_save_path = os.path.join(args.output_dir, "lef_model_checkpoint_last.pt")
        torch.save(lef_model.state_dict(), lef_model_save_path)
        logger.info(f" saved to {args.output_dir}")
        logger.info(f"LEF model weights saved to {lef_model_save_path}")                    
    accelerator.end_training()


if __name__ == "__main__":
    main()

