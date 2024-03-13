import argparse
import os
from PIL import Image
from PIL.ImageOps import invert

import accelerate
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers.utils import ContextManagers

from diffusers import DDPMScheduler, UNet2DConditionModel, VQModel, KandinskyV22PriorPipeline
from diffusers.pipelines.kandinsky2_2.pipline_kandinsky2_2_controlnet_inpainting import KandinskyV22ControlnetInpaintPipeline
from diffusers.training_utils import compute_snr

import wandb

def log_validation(image_encoder, movq, unet, accelerator, weight_dtype, args):
    image_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ]
    )

    prior_pipeline = KandinskyV22PriorPipeline.from_pretrained(
        args.prior_model_path, 
        image_encoder=image_encoder,
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    pipeline = KandinskyV22ControlnetInpaintPipeline.from_pretrained(
        args.decoder_model_path,
        unet=accelerator.unwrap_model(unet),
        movq=movq,
        torch_dtype=weight_dtype,
    )

    prior_pipeline = prior_pipeline.to(accelerator.device)
    prior_pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_logs = []
    for val_prompt, val_image, val_mask, val_cond in zip(args.val_prompts, args.val_images, args.val_masks, args.val_conds):
        val_image = Image.open(val_image).convert("RGB")
        val_mask = Image.open(val_mask).convert("L")
        val_cond = Image.open(val_cond).convert("RGB")

        if args.invert_mask == True:
            val_mask = invert(val_mask)

        with torch.autocast("cuda", torch.float16):
            prior_output = prior_pipeline(
                prompt=val_prompt, 
                negative_prompt="low quality, worst quality, wrinkled, deformed, distorted, jpeg artifacts,nsfw, paintings, sketches, text, watermark, username, spikey",
                generator=generator
            )
            image = pipeline(image=val_image,
                            mask_image=val_mask,
                            hint=image_transforms(val_cond).unsqueeze(0),
                            **prior_output, 
                            height=512,
                            width=512, 
                            num_inference_steps=50,
                            strength=1.0,
                            guidance_scale=4.0,
                            generator=generator).images[0]

        image_logs.append(
            {"images": image, "validation_prompts": val_prompt, "validation_images": val_image, "validation_masks": val_mask, "validation_control": val_cond}
        )
        
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                image = log["images"]
                validation_prompt = log["validation_prompts"]
                validation_image = log["validation_images"]
                validation_mask = log["validation_masks"]
                validation_control = log["validation_control"]
                formatted_images.append(wandb.Image(validation_image, caption=validation_prompt))
                formatted_images.append(wandb.Image(validation_mask, caption="mask"))
                formatted_images.append(wandb.Image(validation_control, caption="object"))
                formatted_images.append(wandb.Image(image, caption="result"))

            tracker.log({"validation": formatted_images})

    del prior_pipeline
    del pipeline
    torch.cuda.empty_cache()

    return True

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of finetuning Kandinsky 2.2.")
    parser.add_argument(
        "--decoder_model_path",
        type=str,
        default="kandinsky-community/kandinsky-2-2-decoder",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prior_model_path",
        type=str,
        default="kandinsky-community/kandinsky-2-2-prior",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./training",
        required=False,
    )
    parser.add_argument(
        "--val_prompts",
        type=str,
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--val_images",
        type=str,
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--val_masks",
        type=str,
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--val_conds",
        type=str,
        default=None,
        nargs="+",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate",
    )
    parser.add_argument(
        "--use_lr_scheduler",
        action="store_true"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--save_ckpt_step",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--validation_step",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--invert_mask",
        action="store_true",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
    )

    args = parser.parse_args()
    return args

from torchvision import transforms
def make_train_dataset(path, image_processor, accelerator, args):
    dataset = load_dataset(path)
    column_names = dataset['train'].column_names
    image_column, conditioning_image_column, mask_column, caption_column = column_names

    image_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.4], [0.3]),
        ]
    )

    mask_transforms = transforms.Compose(
        [
            transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        clip_images = [image.convert("RGB") for image in examples[image_column]]
        clip_images = [image_processor(image, return_tensors="pt").pixel_values[0] for image in clip_images]

        conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
        conditioning_images = [image_transforms(image) for image in conditioning_images]

        masks = [
            invert(mask.convert("L")) 
            if args.invert_mask==True else mask.convert("L") 
            for mask in examples[mask_column]
        ]
        masks = [mask_transforms(mask) for mask in masks]

        examples["pixel_values"] = images
        examples["clip_pixel_values"] = clip_images
        examples["condition_values"] = conditioning_images
        examples["mask_values"] = masks

        return examples
    
    with accelerator.main_process_first():
        train_dataset = dataset["train"].with_transform(preprocess_train)
    
    return train_dataset

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    clip_pixel_values = torch.stack([example["clip_pixel_values"] for example in examples])
    clip_pixel_values = clip_pixel_values.to(memory_format=torch.contiguous_format).float()
    
    condition_values = torch.stack([example["condition_values"] for example in examples])
    condition_values = condition_values.to(memory_format=torch.contiguous_format).float()
    
    mask_values = torch.stack([example["mask_values"] for example in examples])
    mask_values = mask_values.to(memory_format=torch.contiguous_format).float()
    
    return {"pixel_values": pixel_values, "clip_pixel_values": clip_pixel_values,
            "condition_values": condition_values, "mask_values": mask_values}



def main():
    args = parse_args()

    cur_dir = os.path.dirname(os.path.abspath(__name__))
    os.makedirs(os.path.join(cur_dir, "training"), exist_ok=True)
    os.makedirs(os.path.join(cur_dir, "training", "log"), exist_ok=True)
    
    accelerator_project_config = ProjectConfiguration(
        project_dir=os.path.join(cur_dir, "training"),
        logging_dir=os.path.join(cur_dir, "training", "log")
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)

    noise_scheduler = DDPMScheduler.from_pretrained(args.decoder_model_path, subfolder="scheduler")
    image_processor = CLIPImageProcessor.from_pretrained(
        args.prior_model_path, subfolder="image_processor"
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        movq = VQModel.from_pretrained(
            args.decoder_model_path, subfolder="movq", torch_dtype=weight_dtype
        ).eval()
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.prior_model_path, subfolder="image_encoder", torch_dtype=weight_dtype
        ).eval()

    if args.init == True:
        unet = UNet2DConditionModel.from_config(os.path.join(args.decoder_model_path, "unet/config.json"))
    else:
        unet = UNet2DConditionModel.from_pretrained(args.decoder_model_path, subfolder="unet")
        print(f"UNET loaded from {os.path.join(args.decoder_model_path, 'unet')}")
    
    movq.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.train()

    train_dataset = make_train_dataset(args.train_data_dir,
                                       image_processor=image_processor,
                                       accelerator=accelerator,
                                       args=args)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=0,
    )
    
    if args.use_lr_scheduler:
        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=2e-06,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08,
        )
        from lr_scheduler.cosine_base import CosineAnnealingWarmUpRestarts
        lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10000, T_mult=1, eta_max=args.lr,  T_up=100, gamma=1)
    else:
        optimizer = torch.optim.AdamW(
                unet.parameters(),
                lr=args.lr,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
                eps=1e-08,
            )
        from torch.optim.lr_scheduler import LambdaLR
        lr_scheduler = LambdaLR(optimizer, lambda _: 1, last_epoch=-1)


    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers("train_kandinsky_inpaint_hint", config=tracker_config)
    

    image_encoder.to(accelerator.device, dtype=weight_dtype)
    movq.to(accelerator.device, dtype=weight_dtype)

    progress_bar = tqdm(
        range(0, len(train_dataset) * args.epochs),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # log_validation(
    #     image_encoder=image_encoder,
    #     movq=movq,
    #     unet=unet,
    #     accelerator=accelerator,
    #     weight_dtype=weight_dtype,
    #     args=args,
    # )
   
    global_step = 0
    for epoch in range(args.epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            images = batch["pixel_values"].to(weight_dtype)
            masks = batch["mask_values"].to(weight_dtype)
            clip_images = batch["clip_pixel_values"].to(weight_dtype)
            hint = batch["condition_values"].to(weight_dtype)

            images = movq.encode(images).latents
            image_embeds = image_encoder(clip_images).image_embeds
            masked_images = images * masks

            bsz = images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=images.device)
            timesteps = timesteps.long()

            noise = torch.randn_like(images)
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            
            latents = torch.cat([noisy_images, masked_images, masks], dim=1)
            
            target = noise

            added_cond_kwargs = {"image_embeds": image_embeds, "hint": hint}
            
            model_pred = unet(sample=latents, 
                              timestep=timesteps, 
                              encoder_hidden_states=None, 
                              added_cond_kwargs=added_cond_kwargs).sample[:, :4]
            
            if args.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = ((loss * masks).sum([1, 2, 3]) / masks.sum([1, 2, 3])).mean()
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                    dim=1
                )[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss * masks
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()
            
            avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
            train_loss += avg_loss.item()
            
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss = 0.0

            if accelerator.is_main_process:
                if global_step % args.save_ckpt_step == 0:
                    if global_step % args.save_ckpt_step == 0:
                        save_path = os.path.join(args.save_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path,exist_ok=True)

                        unet = accelerator.unwrap_model(unet)
                        torch.save(unet.state_dict(), os.path.join(save_path, f"unet_{epoch}.pth"))
                
                if args.val_prompts is not None and global_step % args.validation_step == 0:
                    log_validation(
                        image_encoder=image_encoder,
                        movq=movq,
                        unet=unet,
                        accelerator=accelerator,
                        weight_dtype=weight_dtype,
                        args=args,
                    )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()