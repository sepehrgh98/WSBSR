from omegaconf import OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser, Namespace
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
import os 
from tqdm import tqdm
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
import itertools
from einops import rearrange


from WSBSR.utils.common import instantiate_from_config, load_model_from_url, wavelet_reconstruction

from WSBSR.utils.cond_fn import MSEGuidance, WeightedMSEGuidance
from WSBSR.sampler import SpacedSampler
from WSBSR.sampler import (
    SpacedSampler,
    DDIMSampler,
    DPMSolverSampler,
    EDMSampler,
)

import torch
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODELS = {
    # --------------- stage-1 model weights ---------------
    "bsrnet": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth",
    # the following checkpoint is up-to-date, we use the old version in our paper
    # "swinir_face": "https://github.com/zsyOAOA/DifFace/releases/download/V1.0/General_Face_ffhq512.pth",
    "swinir_face": "https://huggingface.co/lxq007/DiffBIR/resolve/main/face_swinir_v1.ckpt",
    "scunet_psnr": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
    "swinir_general": "https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt",
    "swinir_realesrgan": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/realesrgan_s4_swinir_100k.pth",
    # --------------- pre-trained stable diffusion weights ---------------
    "sd_v2.1": "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt",
    "sd_v2.1_zsnr": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/sd2.1-base-zsnr-laionaes5.ckpt",
    # --------------- IRControlNet weights ---------------
    "v1_face": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_face.pth",
    "v1_general": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_general.pth",
    "v2": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.pth",
    "v2.1": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/DiffBIR_v2.1.pt",
}

@torch.no_grad()
def validate(svid, cldm, diffusion, val_loader, cond_fn, args, cfg, device, global_step, writer):
    svid.eval()
    cldm.eval()

    total_w_loss, total_c_loss, total_total_loss = 0.0, 0.0, 0.0
    num_batches = 0

    for val_batch in val_loader:
        lq, labels, _ = val_batch
        lq, labels = lq.to(device), labels.to(device)
        lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()
        lq = lq.div(255.0).clamp(0, 1)

        # Forward through SVID
        imageLabel, bFilter = svid(lq)
        loss_w = wsddn_loss(imageLabel, labels)

        # Prepare condition image
        _, _, h, w = lq.shape
        new_size = (int(h * cfg.train.upscale), int(w * cfg.train.upscale))
        cond_img = F.interpolate(lq, size=new_size, mode='bicubic', align_corners=False)

        with torch.autocast(args.device, torch.float16 if args.precision == "fp16" else torch.float32):
            sr_sample = apply_cldm(
                cldm,
                diffusion,
                cond_fn,
                cond_img,
                args.steps,
                args.strength,
                args.vae_encoder_tiled,
                args.vae_encoder_tile_size,
                args.vae_decoder_tiled,
                args.vae_decoder_tile_size,
                args.cldm_tiled,
                args.cldm_tile_size,
                args.cldm_tile_stride,
                args.pos_prompt,
                args.neg_prompt,
                args.cfg_scale,
                args.start_point_type,
                args.sampler,
                args.noise_aug,
                args.rescale_cfg,
                args.s_churn,
                args.s_tmin,
                args.s_tmax,
                args.s_noise,
                args.eta,
                args.order,
                device,
            )

        sr_sample = wavelet_reconstruction((sr_sample + 1) / 2, cond_img).clamp(0, 1)
        lr_2 = bFilter(sr_sample)
        loss_c = F.l1_loss(lr_2, cond_img)
        total_loss = cfg.train.lambda_w * loss_w + cfg.train.lambda_c * loss_c

        total_w_loss += loss_w.item()
        total_c_loss += loss_c.item()
        total_total_loss += total_loss.item()
        num_batches += 1

    if num_batches > 0:
        avg_w = total_w_loss / num_batches
        avg_c = total_c_loss / num_batches
        avg_total = total_total_loss / num_batches

        if writer is not None:
            writer.add_scalar("val/loss_w", avg_w, global_step)
            writer.add_scalar("val/loss_c", avg_c, global_step)
            writer.add_scalar("val/total_loss", avg_total, global_step)

        print(f"[VALIDATION] loss_w: {avg_w:.4f}, loss_c: {avg_c:.4f}, total_loss: {avg_total:.4f}")




def pad_to_multiples_of(imgs: torch.Tensor, multiple: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h % multiple == 0 and w % multiple == 0:
        return imgs.clone()
    ph, pw = map(lambda x: (x + multiple - 1) // multiple * multiple - x, (h, w))
    return F.pad(imgs, pad=(0, pw, 0, ph), mode="constant", value=0)


def apply_cldm(
    cldm,
    diffusion,
    cond_fn,
    cond_img: torch.Tensor,
    steps: int,
    strength: float,
    vae_encoder_tiled: bool,
    vae_encoder_tile_size: int,
    vae_decoder_tiled: bool,
    vae_decoder_tile_size: int,
    cldm_tiled: bool,
    cldm_tile_size: int,
    cldm_tile_stride: int,
    pos_prompt: str,
    neg_prompt: str,
    cfg_scale: float,
    start_point_type: str,
    sampler_type: str,
    noise_aug: int,
    rescale_cfg: bool,
    s_churn: float,
    s_tmin: float,
    s_tmax: float,
    s_noise: float,
    eta: float,
    order: int,
    device: str
) -> torch.Tensor:
    bs, _, h0, w0 = cond_img.shape
    # 1. Pad condition image for VAE encoding (scale factor = 8)
    # 1.1 Whether or not tiled inference is used, the input image size for the VAE must be a multiple of 8.
    if not vae_encoder_tiled and not cldm_tiled:
        # For backward capability, pad condition to be multiples of 64
        cond_img = pad_to_multiples_of(cond_img, multiple=64)
    else:
        cond_img = pad_to_multiples_of(cond_img, multiple=8)
    # 1.2 Check vae encoder tile size
    if vae_encoder_tiled and (
        cond_img.size(2) < vae_encoder_tile_size
        or cond_img.size(3) < vae_encoder_tile_size
    ):
        print("[VAE Encoder]: the input size is tiny and unnecessary to tile.")
        vae_encoder_tiled = False
    # 1.3 If tiled inference is used, then the size of each tile also needs to be a multiple of 8.
    if vae_encoder_tiled:
        if vae_encoder_tile_size % 8 != 0:
            raise ValueError("VAE encoder tile size must be a multiple of 8")
    # with VRAMPeakMonitor("encoding condition image"):
    cond = cldm.prepare_condition(
        cond_img,
        [pos_prompt] * bs,
        vae_encoder_tiled,
        vae_encoder_tile_size,
    )
    uncond = cldm.prepare_condition(
        cond_img,
        [neg_prompt] * bs,
        vae_encoder_tiled,
        vae_encoder_tile_size,
    )
    h1, w1 = cond["c_img"].shape[2:]
    # 2. Pad condition latent for U-Net inference (scale factor = 8)
    # 2.1 Check cldm tile size
    if cldm_tiled and (h1 < cldm_tile_size // 8 or w1 < cldm_tile_size // 8):
        print("[Diffusion]: the input size is tiny and unnecessary to tile.")
        cldm_tiled = False
    # 2.2 Pad conditon latent
    if not cldm_tiled:
        # If tiled inference is not used, apply padding directly.
        cond["c_img"] = pad_to_multiples_of(cond["c_img"], multiple=8)
        uncond["c_img"] = pad_to_multiples_of(uncond["c_img"], multiple=8)
    else:
        # If tiled inference is used, then the latent tile size must be a multiple of 8.
        if cldm_tile_size % 64 != 0:
            raise ValueError("Diffusion tile size must be a multiple of 64")
    h2, w2 = cond["c_img"].shape[2:]
    # 3. Prepare start point of sampling
    if start_point_type == "cond":
        x_0 = cond["c_img"]
        x_T = diffusion.q_sample(
            x_0,
            torch.full(
                (bs,),
                diffusion.num_timesteps - 1,
                dtype=torch.long,
                device=device,
            ),
            torch.randn(x_0.shape, dtype=torch.float32, device=device),
        )
    else:
        x_T = torch.randn((bs, 4, h2, w2), dtype=torch.float32, device=device)
    # 4. Noise augmentation
    if noise_aug > 0:
        cond["c_img"] = diffusion.q_sample(
            x_start=cond["c_img"],
            t=torch.full(size=(bs,), fill_value=noise_aug, device=device),
            noise=torch.randn_like(cond["c_img"]),
        )
        uncond["c_img"] = cond["c_img"].detach().clone()

    if cond_fn:
        cond_fn.load_target(cond_img * 2 - 1)

    # 5. Set control strength
    control_scales = cldm.control_scales
    cldm.control_scales = [strength] * 13

    # 6. Run sampler
    betas = diffusion.betas
    parameterization = diffusion.parameterization
    if sampler_type == "spaced":
        sampler = SpacedSampler(betas, parameterization, rescale_cfg)
    elif sampler_type == "ddim":
        sampler = DDIMSampler(betas, parameterization, rescale_cfg, eta=0)
    elif sampler_type.startswith("dpm"):
        sampler = DPMSolverSampler(
            betas, parameterization, rescale_cfg, sampler_type
        )
    elif sampler_type.startswith("edm"):
        sampler = EDMSampler(
            betas,
            parameterization,
            rescale_cfg,
            sampler_type,
            s_churn,
            s_tmin,
            s_tmax,
            s_noise,
            eta,
            order,
        )
    else:
        raise NotImplementedError(sampler_type)
    # with VRAMPeakMonitor("sampling"):
    z = sampler.sample(
        model=cldm,
        device=device,
        steps=steps,
        x_size=(bs, 4, h2, w2),
        cond=cond,
        uncond=uncond,
        cfg_scale=cfg_scale,
        tiled=cldm_tiled,
        tile_size=cldm_tile_size // 8,
        tile_stride=cldm_tile_stride // 8,
        x_T=x_T,
        progress=True,
    )
    # Remove padding for U-Net input
    z = z[..., :h1, :w1]

    # 7. Decode generated latents
    if vae_decoder_tiled and (
        h1 < vae_decoder_tile_size // 8 or w1 < vae_decoder_tile_size // 8
    ):
        print("[VAE Decoder]: the input size is tiny and unnecessary to tile.")
        vae_decoder_tiled = False
    # with VRAMPeakMonitor("decoding generated latent"):
    x = cldm.vae_decode(
        z,
        vae_decoder_tiled,
        vae_decoder_tile_size // 8,
    )
    x = x[:, :, :h0, :w0]
    cldm.control_scales = control_scales
    return x


def wsddn_loss(pred, labels, model=None, lambda_reg=1e-4, eps=1e-8):
    term = labels * (pred - 0.5) + 0.5
    log_term = torch.log(term + eps)
    loss = -log_term.sum(dim=1).mean()

    # Add L2 regularization
    if model is not None:
        l2_reg = sum(torch.sum(p ** 2) for p in model.parameters())
        loss += (lambda_reg / 2) * l2_reg

    return loss



def main(args):
    # Setup accelerator:
    print("[INFO] Setup Accelerator...")
    # accelerator = Accelerator(split_batches=True)
    dataloader_config = DataLoaderConfiguration(split_batches=True)
    accelerator = Accelerator(dataloader_config=dataloader_config)
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
    auto_cast_type = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.precision]

    # Setup an experiment folder:
    print("[INFO] Setup Experiment Folder...")
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"[INFO] Experiment directory created at {exp_dir}")


    # Setup data:
    print("[INFO] Setup dataset...")
    dataset = instantiate_from_config(cfg.dataset.train)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )



    # loader = DataLoader(
    #     dataset=dataset,
    #     batch_size=cfg.train.batch_size,
    #     num_workers=cfg.train.num_workers,
    #     shuffle=True,
    #     drop_last=True,
    #     pin_memory=True,
    # )
    if accelerator.is_main_process:
        print(f"[INFO] Dataset contains {len(dataset):,} images")

    sigma_pool = dataset.get_pool()
    cfg.model.svid.params.num_classes = len(sigma_pool)

    # initialize svid
    print("[INFO] Setup SVID...")
    svid: SVID = instantiate_from_config(cfg.model.svid)
    svid_ckpt = cfg.train.svid_ckpt
    if svid_ckpt:
        svid = svid.load_state_dict(torch.load(svid_ckpt, map_location="cpu"), strict=True)

        if accelerator.is_local_main_process:
            print(f"[INFO] strictly load weight from checkpoint: {cfg.train.resume}")

    else:
        if accelerator.is_local_main_process:
            print("[INFO] initialize svid from scratch")
    svid.set_sigmaPool(sigma_pool)


    # Create cldm and IRControlNet
    print("[INFO] Setup cldm...")
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    # sd = load_model_from_url(MODELS["sd_v2.1_zsnr"])
    sd = torch.load(cfg.train.cldm_ckpt, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    if accelerator.is_main_process:
        print(
            f"[INFO] strictly load pretrained SD weight from {cfg.train.cldm_ckpt}\n"
            f"unused weights: {unused}\n"
            f"missing weights: {missing}"
        )

    if cfg.train.IRC_ckpt:
        cldm.load_controlnet_from_ckpt(torch.load(cfg.train.IRC_ckpt, map_location="cpu"))
        # control_weight = load_model_from_url(MODELS["v2.1"])
        # cldm.load_controlnet_from_ckpt(control_weight)
        if accelerator.is_main_process:
            print(
                f"[INFO] strictly load controlnet weight from checkpoint: {cfg.train.IRC_ckpt}"
            )
    else:
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        if accelerator.is_main_process:
            print(
                f"[INFO] strictly load controlnet weight from pretrained SD\n"
                f"weights initialized with newly added zeros: {init_with_new_zero}\n"
                f"weights initialized from scratch: {init_with_scratch}"
            )


    # Load Condition Function
    print("[INFO] Load Condition Function...")
    if not args.guidance:
        cond_fn = None
    else:
        if args.g_loss == "mse":
            cond_fn_cls = MSEGuidance
        elif args.g_loss == "w_mse":
            cond_fn_cls = WeightedMSEGuidance
        else:
            raise ValueError(args.g_loss)
        cond_fn = cond_fn_cls(
            args.g_scale,
            args.g_start,
            args.g_stop,
            args.g_space,
            args.g_repeat,
        )


    # Create Diffusion
    print("[INFO] Setup diffusion...")
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)

    # Setup optimizer:
    print("[INFO] Setup optimizer...")
    opt = torch.optim.AdamW(
    itertools.chain(cldm.controlnet.parameters(), svid.parameters()),
    lr=cfg.train.learning_rate)

    # Define loss
    criterion = torch.nn.BCELoss()

    # Use weight_decay in optimizer (built-in L2 regularization)
    optimizer = torch.optim.SGD(
        list(svid.parameters()) + list(cldm.controlnet.parameters()),
        lr=cfg.train.learning_rate,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.lambda_reg  
    )


    # Prepare models for training:
    svid.train().to(device)
    cldm.train().to(device)
    diffusion.to(device)
    cldm, svid, opt, train_loader, val_loader = accelerator.prepare(cldm, svid, opt, train_loader, val_loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    noise_aug_timestep = cfg.train.noise_aug_timestep


    # monitoring/logging initializing
    global_step = 0
    max_steps = cfg.train.train_steps
    step_w_loss = []
    step_c_loss = []
    step_total_loss = []
    epoch = 0
    epoch_w_loss = []
    epoch_c_loss = []
    epoch_total_loss = []


    # Sampler
    sampler = SpacedSampler(
    diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )


    if accelerator.is_main_process:
        logs_dir = os.path.join(exp_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        print(f"[INFO] Logs directory created at {logs_dir}")
        writer = SummaryWriter(logs_dir)
        print(f"[INFO] Training for {max_steps} steps...")


    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_main_process,
            unit="batch",
            total=len(train_loader),
        )

        for batch in train_loader:
            lq, labels, _ = batch
            lq, labels = lq.to(device), labels.to(device)
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()


            lq = (
                    lq.div(255.0)
                    .clamp(0, 1)
                    .contiguous()
                    .to(device=device, dtype=torch.float32)
                )

            
            imageLabel, bFilter = svid(lq) # Image level scores, region-aware bluring kernel

            # SVID Loss
            loss_w  = wsddn_loss(imageLabel, labels)  # SVID loss
            step_w_loss.append(loss_w.item())
            epoch_w_loss.append(loss_w.item())

            if epoch < cfg.train.warmup_epochs:
                # ------- Phase 1: Warmup (Only train SVID) -------
                optimizer.zero_grad()
                loss_w.backward()
                optimizer.step()
                
                continue  # Skip rest of loop
            else:
                # ------- Phase 2: Joint Training -------
                # --- Create Super-Resolved Image ---
                _, _, h, w = lq.shape
                output_size = (h, w)
                new_size = (int(h * cfg.train.upscale), int(w * cfg.train.upscale))
                
                # Bicubic interpolation to upscale
                cond_img = F.interpolate(
                    lq,
                    size=new_size,
                    mode='bicubic',
                    align_corners=False
                )

                cond_img = (
                    cond_img.clamp(0, 1)
                    .contiguous()
                    .to(device=device, dtype=torch.float32)
                )


                # Apply cldm
                torch.cuda.empty_cache()
                with torch.autocast(args.device, auto_cast_type):
                    sr_sample = apply_cldm(
                        cldm,
                        diffusion,
                        cond_fn,
                        cond_img,
                        args.steps,
                        args.strength,
                        args.vae_encoder_tiled,
                        args.vae_encoder_tile_size,
                        args.vae_decoder_tiled,
                        args.vae_decoder_tile_size,
                        args.cldm_tiled,
                        args.cldm_tile_size,
                        args.cldm_tile_stride,
                        args.pos_prompt,
                        args.neg_prompt,
                        args.cfg_scale,
                        args.start_point_type,
                        args.sampler,
                        args.noise_aug,
                        args.rescale_cfg,
                        args.s_churn,
                        args.s_tmin,
                        args.s_tmax,
                        args.s_noise,
                        args.eta,
                        args.order,
                        device,
                    )


                # sr_sample = F.interpolate(
                #     wavelet_reconstruction((sr_sample + 1) / 2, cond_img),
                #     size=output_size,
                #     mode="bicubic",
                #     antialias=True,
                # )
                sr_sample = wavelet_reconstruction((sr_sample + 1) / 2, cond_img).clamp(0, 1)


                lr_2 = bFilter(sr_sample) 

                # Cycle loss
                loss_c = F.l1_loss(lr_2, cond_img)
                step_c_loss.append(loss_c)
                epoch_c_loss.append(loss_c)
                


                # Total loss
                total_loss = cfg.train.lambda_w * loss_w + cfg.train.lambda_c * loss_c


                # Step 9: Backward + Optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                accelerator.wait_for_everyone()

                global_step += 1
                step_total_loss.append(total_loss.item())
                epoch_total_loss.append(total_loss.item())

            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, loss_w: {loss_w.item():.6f}, total_loss: {total_loss.item():.6f}"
            )

            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                # Gather values from all processes
                avg_loss_w = (
                    accelerator.gather(
                        torch.tensor(step_w_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                step_w_loss.clear()

                avg_loss_c = (
                    accelerator.gather(
                        torch.tensor(step_c_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                step_c_loss.clear()

                avg_total_loss = (
                    accelerator.gather(
                        torch.tensor(step_total_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                step_total_loss.clear()



                if accelerator.is_main_process:
                    writer.add_scalar("loss_w/loss_w_simple_step", avg_loss_w, global_step)
                    writer.add_scalar("loss_c/loss_c_simple_step", avg_loss_c, global_step)
                    writer.add_scalar("total_loss/total_loss_simple_step", avg_total_loss, global_step)
            
            

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:

                    #IRControlNet
                    controlnet_checkpoint = pure_cldm.controlnet.state_dict()
                    controlnet_ckpt_path = os.path.join(ckpt_dir, "IRControlNet")
                    os.makedirs(controlnet_ckpt_path, exist_ok=True)
                    controlnet_ckpt_path = f"{controlnet_ckpt_path}/IRControlNet_{global_step:07d}.pt"
                    torch.save(controlnet_checkpoint, controlnet_ckpt_path)

                    # SVID
                    svid_checkpoint = svid.state_dict()
                    svid_ckpt_path = os.path.join(ckpt_dir, "SVID")
                    os.makedirs(svid_ckpt_path, exist_ok=True)
                    svid_ckpt_path = f"{svid_ckpt_path}/SVID_{global_step:07d}.pt"
                    torch.save(svid_checkpoint, svid_ckpt_path)

            if global_step % cfg.train.val_every == 0 and global_step > 0:
                validate(svid, cldm, diffusion, val_loader, cond_fn, args, cfg, device, global_step, writer)
  
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break


        pbar.close()
        epoch += 1
        avg_epoch_loss_w = (
            accelerator.gather(torch.tensor(epoch_w_loss, device=device).unsqueeze(0))
            .mean()
            .item()
        )
        epoch_w_loss.clear()

        avg_epoch_loss_c = (
            accelerator.gather(torch.tensor(epoch_c_loss, device=device).unsqueeze(0))
            .mean()
            .item()
        )
        epoch_c_loss.clear()


        avg_epoch_total_loss = (
            accelerator.gather(torch.tensor(epoch_total_loss, device=device).unsqueeze(0))
            .mean()
            .item()
        )
        epoch_total_loss.clear()

        if accelerator.is_main_process:
            writer.add_scalar("loss_w/loss_w_simple_step", avg_epoch_loss_w, global_step)
            writer.add_scalar("loss_c/loss_c_simple_step", avg_epoch_loss_c, global_step)
            writer.add_scalar("total_loss/total_loss_simple_step", avg_epoch_total_loss, global_step)
    
    if accelerator.is_main_process:
        print("done!")
        writer.close()
                    
            
            


def check_device(device: str) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            print(
                "CUDA not available because the current PyTorch install was not "
                "built with CUDA enabled."
            )
            device = "cpu"
    else:
        if device == "mps":
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print(
                        "MPS not available because the current PyTorch install was not "
                        "built with MPS enabled."
                    )
                    device = "cpu"
                else:
                    print(
                        "MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine."
                    )
                    device = "cpu"
    print(f"using device {device}")
    return device


DEFAULT_POS_PROMPT = (
    "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
    "hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, "
    "skin pore detailing, hyper sharpness, perfect without deformations."
)

DEFAULT_NEG_PROMPT = (
    "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, "
    "CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, "
    "signature, jpeg artifacts, deformed, lowres, over-smooth."
)






def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Sampling steps. More steps, more details.",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=1,
        help="Control strength from ControlNet. Less strength, more creative.",
    )
    parser.add_argument(
        "--vae_encoder_tiled",
        action="store_true",
        help="Enable tiled inference for AE encoder, which reduces the GPU memory usage.",
    )
    parser.add_argument(
        "--vae_encoder_tile_size", type=int, default=256, help="Size of each tile."
    )
    parser.add_argument(
        "--vae_decoder_tiled",
        action="store_true",
        help="Enable tiled inference for AE decoder, which reduces the GPU memory usage.",
    )
    parser.add_argument(
        "--vae_decoder_tile_size", type=int, default=256, help="Size of each tile."
    )
    parser.add_argument(
        "--cldm_tiled",
        action="store_true",
        help="Enable tiled sampling, which reduces the GPU memory usage.",
    )
    parser.add_argument(
        "--cldm_tile_size", type=int, default=512, help="Size of each tile."
    )
    parser.add_argument(
        "--cldm_tile_stride", type=int, default=256, help="Stride between tiles."
    )
    parser.add_argument(
        "--pos_prompt",
        type=str,
        default=DEFAULT_POS_PROMPT,
        help=(
            "Descriptive words for 'good image quality'. "
            "It can also describe the things you WANT to appear in the image."
        ),
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default=DEFAULT_NEG_PROMPT,
        help=(
            "Descriptive words for 'bad image quality'. "
            "It can also describe the things you DON'T WANT to appear in the image."
        ),
    )
    parser.add_argument(
        "--cfg_scale", type=float, default=6.0, help="Classifier-free guidance scale."
    )
    parser.add_argument(
        "--start_point_type",
        type=str,
        choices=["noise", "cond"],
        default="noise",
        help=(
            "For DiffBIR v1 and v2, setting the start point types to 'cond' can make the results much more stable "
            "and ensure that the outcomes from ODE samplers like DDIM and DPMS are normal. "
            "However, this adjustment may lead to a decrease in sample quality."
        ),
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="edm_dpm++_3m_sde",
        choices=[
            "dpm++_m2",
            "spaced",
            "ddim",
            "edm_euler",
            "edm_euler_a",
            "edm_heun",
            "edm_dpm_2",
            "edm_dpm_2_a",
            "edm_lms",
            "edm_dpm++_2s_a",
            "edm_dpm++_sde",
            "edm_dpm++_2m",
            "edm_dpm++_2m_sde",
            "edm_dpm++_3m_sde",
        ],
        help="Sampler type. Different samplers may produce very different samples.",
    )
    parser.add_argument(
        "--noise_aug",
        type=int,
        default=0,
        help="Level of noise augmentation. More noise, more creative.",
    )
    parser.add_argument(
        "--rescale_cfg",
        action="store_true",
        help="Gradually increase cfg scale from 1 to ('cfg_scale' + 1)",
    )
    parser.add_argument(
        "--s_churn",
        type=float,
        default=0,
        help="Randomness in sampling. Only works with some edm samplers.",
    )
    parser.add_argument(
        "--s_tmin",
        type=float,
        default=0,
        help="Minimum sigma for adding ramdomness to sampling. Only works with some edm samplers.",
    )
    parser.add_argument(
        "--s_tmax",
        type=float,
        default=300,
        help="Maximum  sigma for adding ramdomness to sampling. Only works with some edm samplers.",
    )
    parser.add_argument(
        "--s_noise",
        type=float,
        default=1,
        help="Randomness in sampling. Only works with some edm samplers.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1,
        help="I don't understand this parameter. Leave it as default.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=1,
        help="Order of solver. Only works with edm_lms sampler.",
    )
    parser.add_argument(
        "--guidance", action="store_true", help="Enable restoration guidance."
    )
    parser.add_argument(
        "--g_loss",
        type=str,
        default="w_mse",
        choices=["mse", "w_mse"],
        help="Loss function of restoration guidance.",
    )
    parser.add_argument(
        "--g_scale",
        type=float,
        default=0.0,
        help="Learning rate of optimizing the guidance loss function.",
    )
    parser.add_argument(
        "--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"]
    )
    parser.add_argument("--g_start", type=int, default=1001)
    parser.add_argument("--g_stop", type=int, default=-1)
    parser.add_argument("--g_space", type=str, default="latent")
    parser.add_argument("--g_repeat", type=int, default=1)
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"]
    )
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--config", type=str, required=True)

    return parser.parse_args()


        


if __name__ == "__main__":
    args = parse_args()
    print(args.device)
    args.device = check_device(args.device)
    set_seed(args.seed)
    main(args)