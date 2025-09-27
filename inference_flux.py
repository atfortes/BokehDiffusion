import os
import torch
import argparse
import numpy as np
from typing import Optional
from utils import prompt_label
from bokeh_diffusion.utils import color_transfer_lab
from bokeh_diffusion.adapter_flux import BokehFluxControlAdapter
from transformers import CLIPTokenizer, CLIPTextModel, T5TokenizerFast, T5EncoderModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, randn_tensor, calculate_shift


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MIN_BOKEH, MAX_BOKEH = 0, 30


def _encode_prompt_with_t5(
    text_encoder: T5EncoderModel,
    tokenizer: T5TokenizerFast,
    prompt: list[str],
    num_images_per_prompt: int = 1,
    device: torch.device = None
):
    text_input_ids = tokenizer(prompt, padding="max_length", max_length=512, truncation=True, return_length=False,
                               return_overflowing_tokens=False, return_tensors="pt").input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(len(prompt) * num_images_per_prompt, seq_len, -1)
    return prompt_embeds

def _encode_prompt_with_clip(
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    prompt: list[str],
    num_images_per_prompt: int = 1,
    device: torch.device = None
):
    text_inputs_ids = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_overflowing_tokens=False,
                               return_length=False, return_tensors="pt").input_ids
    prompt_embeds = text_encoder(text_inputs_ids.to(device), output_hidden_states=False)

    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(len(prompt) * num_images_per_prompt, -1)
    return prompt_embeds

def encode_prompt(
    text_encoders: list[CLIPTextModel, T5EncoderModel],
    tokenizers: list[CLIPTokenizer, T5TokenizerFast],
    prompt: list[str],
    device: torch.device = None,
    num_images_per_prompt: int = 1
):
    dtype = text_encoders[0].dtype
    device = device if device is not None else text_encoders[1].device
    pooled_prompt_embeds = _encode_prompt_with_clip(text_encoder=text_encoders[0], tokenizer=tokenizers[0],
                                                    prompt=prompt, num_images_per_prompt=num_images_per_prompt, device=device)

    prompt_embeds = _encode_prompt_with_t5(text_encoder=text_encoders[1], tokenizer=tokenizers[1],
                                           prompt=prompt, num_images_per_prompt=num_images_per_prompt, device=device)

    text_ids = torch.zeros(len(prompt), prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)
    return prompt_embeds, pooled_prompt_embeds, text_ids


def generate_image(
    pipeline: FluxPipeline,
    bokeh_adapter: BokehFluxControlAdapter,
    prompt: str,
    seed: int,
    bokeh_target: float,
    guidance_scale: float,
    true_cfg: float,
    num_inference_steps: int,
    height: int = 512,
    width: int = 512,
    num_grounding_steps: int = 0,
    bokeh_pivot: Optional[float] = None,
):
    is_grounded = num_grounding_steps > 0 and bokeh_pivot is not None
    num_images = 2 if is_grounded else 1

    # Encode prompts
    generator = torch.Generator(device="cpu").manual_seed(seed)
    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
        [pipeline.text_encoder, pipeline.text_encoder_2],
        [pipeline.tokenizer, pipeline.tokenizer_2],
        [prompt],
        device=pipeline.device,
        num_images_per_prompt=num_images,
    )
    neg_prompt_embeds, pooled_neg_prompt_embeds, _ = encode_prompt(
        [pipeline.text_encoder, pipeline.text_encoder_2],
        [pipeline.tokenizer, pipeline.tokenizer_2],
        ["bad quality, low resolution"],
        device=pipeline.device,
        num_images_per_prompt=num_images,
    )

    # Bokeh annotation
    if is_grounded:
        bokeh_ann = torch.tensor([bokeh_target, bokeh_pivot], dtype=pipeline.vae.dtype, device=pipeline.device)
    else:
        bokeh_ann = torch.tensor([bokeh_target], dtype=pipeline.vae.dtype, device=pipeline.device)
    bokeh_ann = bokeh_ann.unsqueeze(1)
    neg_bokeh_ann = torch.full_like(bokeh_ann, -1)

    # Latent init
    num_ch = pipeline.transformer.config.in_channels // 4
    h_lat = 2 * (height // (pipeline.vae_scale_factor * 2))
    w_lat = 2 * (width // (pipeline.vae_scale_factor * 2))
    latents = randn_tensor((1, num_ch, h_lat, w_lat), generator=generator, device=pipeline.device, dtype=pipeline.vae.dtype).expand(num_images, -1, -1, -1)
    latents = FluxPipeline._pack_latents(latents, num_images, num_ch, h_lat, w_lat)
    latent_img_ids = FluxPipeline._prepare_latent_image_ids(num_images, h_lat // 2, w_lat // 2, pipeline.device, prompt_embeds.dtype)

    # Timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    mu = calculate_shift(latents.shape[1], 256, 4096, 0.5, 1.15)
    timesteps, _ = retrieve_timesteps(pipeline.scheduler, num_inference_steps, pipeline.device, sigmas=sigmas, mu=mu)

    guidance = None
    if pipeline.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=pipeline.device, dtype=torch.float32).expand(latents.shape[0])

    for i, t in enumerate(timesteps):
        pipeline._current_timestep = t
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        # Positive branch
        noise_pred = bokeh_adapter(
            pipeline.transformer,
            bokeh_ann,
            perform_swap=(i < num_grounding_steps),
            batch_swap_ids=[1, 1] if num_grounding_steps > 0 else None,
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids[0],
            img_ids=latent_img_ids,
        )
        if true_cfg > 1.0:
            # Negative branch
            noise_neg = bokeh_adapter(
                pipeline.transformer,
                neg_bokeh_ann,
                perform_swap=False,
                batch_swap_ids=None,
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=torch.ones_like(guidance),
                pooled_projections=pooled_neg_prompt_embeds,
                encoder_hidden_states=neg_prompt_embeds,
                txt_ids=text_ids[0],
                img_ids=latent_img_ids,
            )
            noise_pred = noise_neg + true_cfg * (noise_pred - noise_neg)
        
        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    pipeline._current_timestep = None

    # Decode
    latents = FluxPipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
    latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    image = pipeline.vae.decode(latents.to(dtype=pipeline.vae.dtype), return_dict=False)[0]
    image = pipeline.image_processor.postprocess(image, output_type="pil")

    if is_grounded:
        image[0] = color_transfer_lab(image[1], image[0], adjust_std=False)
    return image[0]


def main(args):
    os.makedirs(args.results_dir, exist_ok=True)

    # FLUX Pipeline
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev"
    ).to(DEVICE, dtype=torch.bfloat16)

    # Bokeh Adapter
    bokeh_adapter = BokehFluxControlAdapter.from_pretrained(
        "atfortes/BokehDiffusion",
        base_model=pipeline.transformer,
    ).to(DEVICE, dtype=torch.bfloat16)

    # Generation
    for bokeh_target in args.bokeh_targets:
        fname = f"{prompt_label(args.prompt)}-bokeh{bokeh_target:.0f}.jpg"
        print(f"Generating {fname}...")
        with torch.no_grad():
            img = generate_image(
                pipeline,
                bokeh_adapter,
                args.prompt,
                args.seed,
                bokeh_target / MAX_BOKEH,
                args.guidance_scale,
                args.true_cfg,
                args.num_inference_steps,
                args.height,
                args.width,
                args.num_grounding_steps,
                args.bokeh_pivot / MAX_BOKEH if args.bokeh_pivot is not None else None,
            )
        img.save(os.path.join(args.results_dir, fname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flux inference parameters')
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--bokeh_targets', nargs='+', type=float)  # 0-30 floats, target images
    parser.add_argument('--seed', type=int, default=65578)
    parser.add_argument('--bokeh_pivot', type=float, default=None, required=False)  # 10.0-20.0 recommended (grounded generation)
    parser.add_argument('--num_inference_steps', type=int, default=30)  # 30 recommended
    parser.add_argument('--num_grounding_steps', type=int, default=0)  # 18-24 recommended (grounded generation)
    parser.add_argument('--guidance_scale', type=float, default=3.0)  # 3.0-6.0 recommended
    parser.add_argument('--true_cfg', type=float, default=1.0)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--results_dir', type=str, default="results")
    args = parser.parse_args()
    main(args)
