import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections import namedtuple

import comfy.utils
import comfy.model_management
import folder_paths

# Model Definitions (from madebyollin/taesd and madebyollin/taehv)


# TAESD (for images)
def taesd_conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class TAESD_Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class TAESD_Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            taesd_conv(n_in, n_out),
            nn.ReLU(),
            taesd_conv(n_out, n_out),
            nn.ReLU(),
            taesd_conv(n_out, n_out),
        )
        self.skip = (
            nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        )
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


def TAESD_Encoder(latent_channels=4):
    return nn.Sequential(
        taesd_conv(3, 64),
        TAESD_Block(64, 64),
        taesd_conv(64, 64, stride=2, bias=False),
        TAESD_Block(64, 64),
        TAESD_Block(64, 64),
        TAESD_Block(64, 64),
        taesd_conv(64, 64, stride=2, bias=False),
        TAESD_Block(64, 64),
        TAESD_Block(64, 64),
        TAESD_Block(64, 64),
        taesd_conv(64, 64, stride=2, bias=False),
        TAESD_Block(64, 64),
        TAESD_Block(64, 64),
        TAESD_Block(64, 64),
        taesd_conv(64, latent_channels),
    )


def TAESD_Decoder(latent_channels=4):
    return nn.Sequential(
        TAESD_Clamp(),
        taesd_conv(latent_channels, 64),
        nn.ReLU(),
        TAESD_Block(64, 64),
        TAESD_Block(64, 64),
        TAESD_Block(64, 64),
        nn.Upsample(scale_factor=2),
        taesd_conv(64, 64, bias=False),
        TAESD_Block(64, 64),
        TAESD_Block(64, 64),
        TAESD_Block(64, 64),
        nn.Upsample(scale_factor=2),
        taesd_conv(64, 64, bias=False),
        TAESD_Block(64, 64),
        TAESD_Block(64, 64),
        TAESD_Block(64, 64),
        nn.Upsample(scale_factor=2),
        taesd_conv(64, 64, bias=False),
        TAESD_Block(64, 64),
        taesd_conv(64, 3),
    )


class TAESD(nn.Module):
    def __init__(self, encoder_path=None, decoder_path=None, latent_channels=None):
        super().__init__()
        if latent_channels is None:
            latent_channels = self.guess_latent_channels(str(encoder_path))
        self.encoder = TAESD_Encoder(latent_channels)
        self.decoder = TAESD_Decoder(latent_channels)
        if encoder_path is not None:
            self.encoder.load_state_dict(comfy.utils.load_torch_file(encoder_path))
        if decoder_path is not None:
            self.decoder.load_state_dict(comfy.utils.load_torch_file(decoder_path))

    def guess_latent_channels(self, encoder_path):
        if "taef1" in encoder_path or "taesd3" in encoder_path:
            return 16
        return 4


# TAEHV (for video)
TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))


def taehv_conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class TAEHV_Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class TAEHV_MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            taehv_conv(n_in * 2, n_out),
            nn.ReLU(inplace=True),
            taehv_conv(n_out, n_out),
            nn.ReLU(inplace=True),
            taehv_conv(n_out, n_out),
        )
        self.skip = (
            nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TAEHV_TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


class TAEHV_TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


def apply_model_with_memblocks(model, x, parallel, show_progress_bar):
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    if parallel:
        x = x.reshape(N * T, C, H, W)
        for b in tqdm(model, disable=not show_progress_bar):
            if isinstance(b, TAEHV_MemBlock):
                NT, C, H, W = x.shape
                T = NT // N
                _x = x.reshape(N, T, C, H, W)
                mem = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T].reshape(
                    x.shape
                )
                x = b(x, mem)
            else:
                x = b(x)
        NT, C, H, W = x.shape
        T = NT // N
        x = x.view(N, T, C, H, W)
    else:
        out = []
        work_queue = [
            TWorkItem(xt, 0)
            for t, xt in enumerate(x.reshape(N, T * C, H, W).chunk(T, dim=1))
        ]
        progress_bar = tqdm(range(T), disable=not show_progress_bar)
        mem = [None] * len(model)
        while work_queue:
            xt, i = work_queue.pop(0)
            if i == 0:
                progress_bar.update(1)
            if i == len(model):
                out.append(xt)
            else:
                b = model[i]
                if isinstance(b, TAEHV_MemBlock):
                    if mem[i] is None:
                        xt_new = b(xt, xt * 0)
                        mem[i] = xt
                    else:
                        xt_new = b(xt, mem[i])
                        mem[i].copy_(xt)
                    work_queue.insert(0, TWorkItem(xt_new, i + 1))
                elif isinstance(b, TAEHV_TPool):
                    if mem[i] is None:
                        mem[i] = []
                    mem[i].append(xt)
                    if len(mem[i]) > b.stride:
                        raise ValueError("TAEHV TPool error")
                    elif len(mem[i]) < b.stride:
                        pass
                    else:
                        N, C, H, W = xt.shape
                        xt = b(torch.cat(mem[i], 1).view(N * b.stride, C, H, W))
                        mem[i] = []
                        work_queue.insert(0, TWorkItem(xt, i + 1))
                elif isinstance(b, TAEHV_TGrow):
                    xt = b(xt)
                    NT, C, H, W = xt.shape
                    for xt_next in reversed(
                        xt.view(N, b.stride * C, H, W).chunk(b.stride, 1)
                    ):
                        work_queue.insert(0, TWorkItem(xt_next, i + 1))
                else:
                    xt = b(xt)
                    work_queue.insert(0, TWorkItem(xt, i + 1))
        progress_bar.close()
        x = torch.stack(out, 1)
    return x


class TAEHV(nn.Module):
    def __init__(
        self,
        checkpoint_path="taehv.pth",
        decoder_time_upscale=(True, True),
        decoder_space_upscale=(True, True, True),
        patch_size=1,
        latent_channels=16,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.image_channels = 3
        if checkpoint_path is not None and "taew2_2" in checkpoint_path:
            self.patch_size, self.latent_channels = 2, 48
        self.encoder = nn.Sequential(
            taehv_conv(self.image_channels * self.patch_size**2, 64),
            nn.ReLU(inplace=True),
            TAEHV_TPool(64, 2),
            taehv_conv(64, 64, stride=2, bias=False),
            TAEHV_MemBlock(64, 64),
            TAEHV_MemBlock(64, 64),
            TAEHV_MemBlock(64, 64),
            TAEHV_TPool(64, 2),
            taehv_conv(64, 64, stride=2, bias=False),
            TAEHV_MemBlock(64, 64),
            TAEHV_MemBlock(64, 64),
            TAEHV_MemBlock(64, 64),
            TAEHV_TPool(64, 1),
            taehv_conv(64, 64, stride=2, bias=False),
            TAEHV_MemBlock(64, 64),
            TAEHV_MemBlock(64, 64),
            TAEHV_MemBlock(64, 64),
            taehv_conv(64, self.latent_channels),
        )
        n_f = [256, 128, 64, 64]
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1
        self.decoder = nn.Sequential(
            TAEHV_Clamp(),
            taehv_conv(self.latent_channels, n_f[0]),
            nn.ReLU(inplace=True),
            TAEHV_MemBlock(n_f[0], n_f[0]),
            TAEHV_MemBlock(n_f[0], n_f[0]),
            TAEHV_MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TAEHV_TGrow(n_f[0], 1),
            taehv_conv(n_f[0], n_f[1], bias=False),
            TAEHV_MemBlock(n_f[1], n_f[1]),
            TAEHV_MemBlock(n_f[1], n_f[1]),
            TAEHV_MemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            TAEHV_TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
            taehv_conv(n_f[1], n_f[2], bias=False),
            TAEHV_MemBlock(n_f[2], n_f[2]),
            TAEHV_MemBlock(n_f[2], n_f[2]),
            TAEHV_MemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            TAEHV_TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
            taehv_conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True),
            taehv_conv(n_f[3], self.image_channels * self.patch_size**2),
        )
        if checkpoint_path is not None:
            self.load_state_dict(
                self.patch_tgrow_layers(comfy.utils.load_torch_file(checkpoint_path))
            )

    def patch_tgrow_layers(self, sd):
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TAEHV_TGrow):
                key = f"decoder.{i}.conv.weight"
                if sd[key].shape[0] > new_sd[key].shape[0]:
                    sd[key] = sd[key][-new_sd[key].shape[0] :]
        return sd

    def encode_video(self, x, parallel=True, show_progress_bar=True):
        if self.patch_size > 1:
            x = F.pixel_unshuffle(x, self.patch_size)
        if x.shape[1] % 4 != 0:
            n_pad = 4 - x.shape[1] % 4
            padding = x[:, -1:].repeat_interleave(n_pad, dim=1)
            x = torch.cat([x, padding], 1)
        return apply_model_with_memblocks(self.encoder, x, parallel, show_progress_bar)

    def decode_video(self, x, parallel=True, show_progress_bar=True):
        x = apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)
        x = x.clamp_(0, 1)
        if self.patch_size > 1:
            x = F.pixel_shuffle(x, self.patch_size)
        return x[:, self.frames_to_trim :]


# Node Implementation

TINYAE_MODELS = {
    "taesd": {
        "type": "taesd",
        "dtype": torch.float32,
        "encoder_path": "taesd_encoder",
        "decoder_path": "taesd_decoder",
        "scale": 0.18215,
        "shift": 0.0,
    },
    "taesdxl": {
        "type": "taesd",
        "dtype": torch.float32,
        "encoder_path": "taesdxl_encoder",
        "decoder_path": "taesdxl_decoder",
        "scale": 0.13025,
        "shift": 0.0,
    },
    "taesd3": {
        "type": "taesd",
        "dtype": torch.float32,
        "encoder_path": "taesd3_encoder",
        "decoder_path": "taesd3_decoder",
        "scale": 1.5305,
        "shift": 0.0609,
    },
    "taef1": {
        "type": "taesd",
        "dtype": torch.float32,
        "encoder_path": "taef1_encoder",
        "decoder_path": "taef1_decoder",
        "scale": 0.3611,
        "shift": 0.1159,
    },
    "taew2_1": {"type": "taehv", "dtype": torch.float16, "model_path": "taew2_1"},
    "taew2_2": {"type": "taehv", "dtype": torch.float16, "model_path": "taew2_2"},
}
_TINYAE_CACHE = {}


def get_tinyme_model(model_name):
    if model_name in _TINYAE_CACHE:
        return _TINYAE_CACHE[model_name]

    if model_name not in TINYAE_MODELS:
        raise ValueError(f"Unknown TinyAE model: {model_name}")

    config = TINYAE_MODELS[model_name]
    model_type = config["type"]
    device = comfy.model_management.get_torch_device()
    dtype = config.get("dtype", torch.float32)

    model = None
    if model_type == "taesd":
        encoder_path = folder_paths.get_full_path(
            "vae_approx", f'{config["encoder_path"]}.pth'
        ) or folder_paths.get_full_path(
            "vae_approx", f'{config["encoder_path"]}.safetensors'
        )
        decoder_path = folder_paths.get_full_path(
            "vae_approx", f'{config["decoder_path"]}.pth'
        ) or folder_paths.get_full_path(
            "vae_approx", f'{config["decoder_path"]}.safetensors'
        )
        if not encoder_path or not decoder_path:
            raise FileNotFoundError(
                f"Could not find encoder/decoder for {model_name} in models/vae_approx"
            )
        model = TAESD(encoder_path, decoder_path).to(device, dtype)
    elif model_type == "taehv":
        model_path = folder_paths.get_full_path(
            "vae_approx", f'{config["model_path"]}.pth'
        ) or folder_paths.get_full_path(
            "vae_approx", f'{config["model_path"]}.safetensors'
        )
        if not model_path:
            raise FileNotFoundError(
                f"Could not find model for {model_name} in models/vae_approx"
            )
        model = TAEHV(checkpoint_path=model_path).to(device, dtype)

    if model:
        model.eval()
        _TINYAE_CACHE[model_name] = model
        return model

    return None


def list_available_tinyme_models():
    models = []
    try:
        vae_approx_path = folder_paths.get_folder_paths("vae_approx")[0]
        if not os.path.isdir(vae_approx_path):
            return models
    except IndexError:
        return models  # vae_approx folder doesn't exist

    for name, config in TINYAE_MODELS.items():
        try:
            if config["type"] == "taesd":
                encoder_file_pth = os.path.join(
                    vae_approx_path, f'{config["encoder_path"]}.pth'
                )
                decoder_file_pth = os.path.join(
                    vae_approx_path, f'{config["decoder_path"]}.pth'
                )
                encoder_file_st = os.path.join(
                    vae_approx_path, f'{config["encoder_path"]}.safetensors'
                )
                decoder_file_st = os.path.join(
                    vae_approx_path, f'{config["decoder_path"]}.safetensors'
                )
                if (
                    os.path.exists(encoder_file_pth)
                    and os.path.exists(decoder_file_pth)
                ) or (
                    os.path.exists(encoder_file_st) and os.path.exists(decoder_file_st)
                ):
                    models.append(name)

            elif config["type"] == "taehv":
                model_file_pth = os.path.join(
                    vae_approx_path, f'{config["model_path"]}.pth'
                )
                model_file_st = os.path.join(
                    vae_approx_path, f'{config["model_path"]}.safetensors'
                )
                if os.path.exists(model_file_pth) or os.path.exists(model_file_st):
                    models.append(name)
        except Exception as e:
            print(f"[TinyAE] Warning: Could not check for TinyAE model {name}: {e}")
    return models


class TinyAE_Encode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tinyme_model": (list_available_tinyme_models(),),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "latent/tiny_ae"

    def encode(self, image, tinyme_model):
        model = get_tinyme_model(tinyme_model)
        config = TINYAE_MODELS[tinyme_model]
        device = comfy.model_management.get_torch_device()
        dtype = config.get("dtype", torch.float32)

        pixels = image.to(device, dtype).permute(0, 3, 1, 2)  # BHWC -> BCHW

        with torch.no_grad():
            if config["type"] == "taesd":
                latents = torch.cat([model.encoder(p.unsqueeze(0)) for p in pixels])
                latents = (latents - config["shift"]) / config["scale"]
            elif config["type"] == "taehv":
                # TAEHV expects 5D [Batch, Time, Channels, Height, Width] (NTCHW)
                pixels_5d = pixels.unsqueeze(0)
                # encode_video returns NTCHW: [1, T, C, H, W]
                latents_5d = model.encode_video(pixels_5d, show_progress_bar=False)
                # Convert to ComfyUI video latent format: [Batch, Channels, Time, Height, Width]
                latents = latents_5d.permute(0, 2, 1, 3, 4)

        return ({"samples": latents.cpu()},)


class TinyAE_Decode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "tinyme_model": (list_available_tinyme_models(),),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent/tiny_ae"

    def decode(self, samples, tinyme_model):
        model = get_tinyme_model(tinyme_model)
        config = TINYAE_MODELS[tinyme_model]
        device = comfy.model_management.get_torch_device()
        dtype = config.get("dtype", torch.float32)

        latents = samples["samples"].to(device, dtype)

        with torch.no_grad():
            if config["type"] == "taesd":
                latents = (latents * config["scale"]) + config["shift"]
                pixels = torch.cat([model.decoder(l.unsqueeze(0)) for l in latents])
            elif config["type"] == "taehv":
                # TAEHV expects 5D [Batch, Time, Channels, Height, Width] (NTCHW)
                if latents.ndim == 4:
                    # Standard latent batch (frames), add a batch dimension for TAEHV.
                    # [T, C, H, W] -> [1, T, C, H, W]
                    latents_5d = latents.unsqueeze(0)
                elif latents.ndim == 5:
                    # Prob from a video VAE (e.g., Wan), shape [B, C, T, H, W].
                    # Permute to the expected [B, T, C, H, W].
                    latents_5d = latents.permute(0, 2, 1, 3, 4)
                else:
                    raise ValueError(
                        f"TAEHV Decode received unexpected latent dimensions: {latents.ndim}. Expected 4D or 5D."
                    )

                # decode_video returns [B, T, C, H, W]
                pixels_5d = model.decode_video(latents_5d, show_progress_bar=False)
                # Squeeze the batch dimension to get [T, C, H, W] for ComfyUI
                pixels = pixels_5d.squeeze(0)

        # Convert to ComfyUI's IMAGE format [Batch, Height, Width, Channels]
        images = pixels.permute(0, 2, 3, 1).cpu().float()
        return (images,)


class TinyAE_EncodeTiled:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tinyme_model": (
                    [
                        m
                        for m in list_available_tinyme_models()
                        if TINYAE_MODELS[m]["type"] == "taesd"
                    ],
                ),
                "tile_size": (
                    "INT",
                    {"default": 512, "min": 256, "max": 4096, "step": 64},
                ),
                "tile_overlap": (
                    "INT",
                    {"default": 64, "min": 0, "max": 512, "step": 32},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "latent/tiny_ae"

    def encode(self, image, tinyme_model, tile_size, tile_overlap):
        model = get_tinyme_model(tinyme_model)
        config = TINYAE_MODELS[tinyme_model]
        device = comfy.model_management.get_torch_device()
        dtype = config.get("dtype", torch.float32)

        pixels = image.to(device, dtype).permute(0, 3, 1, 2)  # BHWC -> BCHW

        if max(pixels.shape[-2], pixels.shape[-1]) <= tile_size:
            with torch.no_grad():
                latents = torch.cat([model.encoder(p.unsqueeze(0)) for p in pixels])
                latents = (latents - config["shift"]) / config["scale"]
            return ({"samples": latents.cpu()},)

        all_latents = []
        with torch.no_grad():
            for p in pixels:
                tiled_latents = comfy.utils.tiled_encode(
                    p.unsqueeze(0),
                    model.encoder,
                    tile_x=tile_size,
                    tile_y=tile_size,
                    overlap=tile_overlap,
                )
                all_latents.append(tiled_latents)
            latents = torch.cat(all_latents)
            latents = (latents - config["shift"]) / config["scale"]

        return ({"samples": latents.cpu()},)


class TinyAE_DecodeTiled:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "tinyme_model": (
                    [
                        m
                        for m in list_available_tinyme_models()
                        if TINYAE_MODELS[m]["type"] == "taesd"
                    ],
                ),
                "tile_size": (
                    "INT",
                    {"default": 64, "min": 32, "max": 512, "step": 32},
                ),
                "tile_overlap": ("INT", {"default": 8, "min": 0, "max": 64, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent/tiny_ae"

    def decode(self, samples, tinyme_model, tile_size, tile_overlap):
        print(
            "Note: Tiled decoding with TinyAE may produce visible seams due to its limited receptive field."
        )
        model = get_tinyme_model(tinyme_model)
        config = TINYAE_MODELS[tinyme_model]
        device = comfy.model_management.get_torch_device()
        dtype = config.get("dtype", torch.float32)

        latents = samples["samples"].to(device, dtype)
        latents = (latents * config["scale"]) + config["shift"]

        if max(latents.shape[-2], latents.shape[-1]) <= tile_size:
            with torch.no_grad():
                pixels = torch.cat([model.decoder(l.unsqueeze(0)) for l in latents])
            images = pixels.permute(0, 2, 3, 1).cpu().float()
            return (images,)

        all_pixels = []
        with torch.no_grad():
            for l in latents:
                tiled_pixels = comfy.utils.tiled_decode(
                    l.unsqueeze(0),
                    model.decoder,
                    tile_x=tile_size,
                    tile_y=tile_size,
                    overlap=tile_overlap,
                )
                all_pixels.append(tiled_pixels)
            pixels = torch.cat(all_pixels)

        images = pixels.permute(0, 2, 3, 1).cpu().float()
        return (images,)


NODE_CLASS_MAPPINGS = {
    "TinyAE_Encode": TinyAE_Encode,
    "TinyAE_Decode": TinyAE_Decode,
    "TinyAE_EncodeTiled": TinyAE_EncodeTiled,
    "TinyAE_DecodeTiled": TinyAE_DecodeTiled,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TinyAE_Encode": "TinyAE Encode (Image/Video)",
    "TinyAE_Decode": "TinyAE Decode (Image/Video)",
    "TinyAE_EncodeTiled": "TinyAE Encode Tiled (Image)",
    "TinyAE_DecodeTiled": "TinyAE Decode Tiled (Image)",
}

