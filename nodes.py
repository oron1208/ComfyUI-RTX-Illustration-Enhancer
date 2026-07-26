import math

import torch
import torch.nn.functional as F


def _to_bchw(image):
    return image.movedim(-1, 1).float()


def _to_bhwc(image):
    return image.movedim(1, -1).clamp(0.0, 1.0)


def _gaussian_blur(x, radius):
    radius = max(0, int(radius))
    if radius == 0:
        return x
    sigma = max(radius / 2.5, 0.5)
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    channels = x.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    x = F.pad(x, (radius, radius, 0, 0), mode="replicate")
    x = F.conv2d(x, horizontal, groups=channels)
    x = F.pad(x, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(x, vertical, groups=channels)


def _normalize_depth(depth, batch, height, width, device, dtype):
    if depth is None:
        return None
    if depth.ndim == 4:
        if depth.shape[-1] in (1, 3, 4):
            depth = depth[..., :3].mean(dim=-1)
        elif depth.shape[1] in (1, 3, 4):
            depth = depth[:, :3].mean(dim=1)
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    if depth.shape[0] == 1 and batch > 1:
        depth = depth.expand(batch, -1, -1)
    depth = depth[:batch].unsqueeze(1).to(device=device, dtype=dtype)
    depth = F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
    lo = depth.amin(dim=(2, 3), keepdim=True)
    hi = depth.amax(dim=(2, 3), keepdim=True)
    return (depth - lo) / (hi - lo + 1e-6)


def _mask_tensor(mask, batch, height, width, device, dtype):
    if mask is None:
        return None
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4:
        mask = mask[..., 0] if mask.shape[-1] <= 4 else mask[:, 0]
    if mask.shape[0] == 1 and batch > 1:
        mask = mask.expand(batch, -1, -1)
    mask = mask[:batch].unsqueeze(1).to(device=device, dtype=dtype)
    return F.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False).clamp(0, 1)


def _hex_color(value, device, dtype):
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    try:
        rgb = [int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    except (ValueError, IndexError):
        rgb = [1.0, 0.87, 0.68]
    return torch.tensor(rgb, device=device, dtype=dtype).view(1, 3, 1, 1)


def _normal_from_depth(depth, strength):
    dx = F.pad(depth[:, :, :, 2:] - depth[:, :, :, :-2], (1, 1, 0, 0), mode="replicate")
    dy = F.pad(depth[:, :, 2:, :] - depth[:, :, :-2, :], (0, 0, 1, 1), mode="replicate")
    nx = -dx * strength
    ny = -dy * strength
    nz = torch.ones_like(nx)
    return F.normalize(torch.cat((nx, ny, nz), dim=1), dim=1, eps=1e-6)


class RTXIllustrationEnhancer:
    PRESETS = ["custom", "subtle", "anime_luxury", "cinematic", "jewel_glow", "dramatic"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (cls.PRESETS, {"default": "anime_luxury"}),
                "detail_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01}),
                "lighting_strength": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 2.0, "step": 0.01}),
                "light_azimuth": ("FLOAT", {"default": -45.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "light_elevation": ("FLOAT", {"default": 40.0, "min": 1.0, "max": 89.0, "step": 1.0}),
                "light_color": ("STRING", {"default": "#ffd9ad"}),
                "rim_light": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01}),
                "specular": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 2.0, "step": 0.01}),
                "ambient_occlusion": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bloom": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.5, "step": 0.01}),
                "vibrance": ("FLOAT", {"default": 0.15, "min": -1.0, "max": 1.0, "step": 0.01}),
                "original_preservation": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "depth_strength": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 30.0, "step": 0.1}),
                "invert_depth": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "depth": ("IMAGE",),
                "effect_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("enhanced", "lighting_pass", "normal_map", "depth_preview")
    FUNCTION = "enhance"
    CATEGORY = "RTX Illustration"
    DESCRIPTION = "Fast GPU pseudo-ray-traced relighting and detail enhancement for RTX VSR output."

    @staticmethod
    def _preset_values(name, values):
        presets = {
            "subtle": dict(detail=0.55, light=0.50, rim=0.45, spec=0.55, ao=0.55, bloom=0.45, vibrance=0.55, preserve=1.20),
            "anime_luxury": dict(detail=1.15, light=1.00, rim=1.10, spec=1.10, ao=0.90, bloom=1.10, vibrance=1.15, preserve=0.95),
            "cinematic": dict(detail=0.90, light=1.30, rim=1.35, spec=0.90, ao=1.35, bloom=1.00, vibrance=0.75, preserve=0.88),
            "jewel_glow": dict(detail=1.20, light=1.10, rim=1.60, spec=2.00, ao=0.75, bloom=2.00, vibrance=1.55, preserve=0.78),
            "dramatic": dict(detail=1.05, light=1.65, rim=1.70, spec=1.25, ao=1.75, bloom=1.25, vibrance=0.90, preserve=0.72),
        }
        if name == "custom":
            return values
        multipliers = presets.get(name, presets["anime_luxury"])
        result = {key: values[key] * multipliers[key] for key in values}
        result["preserve"] = min(1.0, result["preserve"])
        return result

    def enhance(
        self, image, preset, detail_strength, lighting_strength, light_azimuth,
        light_elevation, light_color, rim_light, specular, ambient_occlusion,
        bloom, vibrance, original_preservation, depth_strength, invert_depth,
        depth=None, effect_mask=None,
    ):
        source = _to_bchw(image).clamp(0, 1)
        batch, _, height, width = source.shape
        device, dtype = source.device, source.dtype
        values = dict(
            detail=detail_strength, light=lighting_strength, rim=rim_light,
            spec=specular, ao=ambient_occlusion, bloom=bloom,
            vibrance=vibrance, preserve=original_preservation,
        )
        values = self._preset_values(preset, values)

        depth_map = _normalize_depth(depth, batch, height, width, device, dtype)
        if depth_map is None:
            luminance = 0.2126 * source[:, 0:1] + 0.7152 * source[:, 1:2] + 0.0722 * source[:, 2:3]
            large = _gaussian_blur(luminance, max(2, min(height, width) // 96))
            depth_map = (0.65 * large + 0.35 * luminance).clamp(0, 1)
        if invert_depth:
            depth_map = 1.0 - depth_map

        normals = _normal_from_depth(depth_map, depth_strength)
        azimuth = math.radians(light_azimuth)
        elevation = math.radians(light_elevation)
        light_dir = torch.tensor(
            [math.cos(elevation) * math.cos(azimuth),
             math.cos(elevation) * math.sin(azimuth),
             math.sin(elevation)],
            device=device, dtype=dtype,
        ).view(1, 3, 1, 1)
        diffuse = (normals * light_dir).sum(dim=1, keepdim=True).clamp(0, 1)
        light_rgb = _hex_color(light_color, device, dtype)

        ao_radius = max(2, min(height, width) // 80)
        neighborhood = _gaussian_blur(depth_map, ao_radius)
        ao_map = ((neighborhood - depth_map) * 5.0).clamp(0, 1)

        half_vec = F.normalize(light_dir + torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 3, 1, 1), dim=1)
        specular_map = (normals * half_vec).sum(dim=1, keepdim=True).clamp(0, 1).pow(28.0)
        rim_map = (1.0 - normals[:, 2:3].clamp(0, 1)).pow(1.6)
        edge_depth = (depth_map - _gaussian_blur(depth_map, max(1, ao_radius // 2))).abs() * 6.0
        rim_map = (rim_map + edge_depth).clamp(0, 1)

        lighting = (
            1.0
            + values["light"] * (diffuse - 0.45) * 0.85
            - values["ao"] * ao_map * 0.65
        )
        worked = source * lighting
        worked = worked + light_rgb * diffuse * values["light"] * 0.14
        worked = worked + light_rgb * rim_map * values["rim"] * 0.24
        worked = worked + light_rgb * specular_map * values["spec"] * 0.42

        detail_radius = max(1, min(height, width) // 512)
        detail = source - _gaussian_blur(source, detail_radius)
        worked = worked + detail * values["detail"]

        gray = worked.mean(dim=1, keepdim=True)
        worked = gray + (worked - gray) * (1.0 + values["vibrance"])

        threshold = 0.72
        highlights = (worked - threshold).clamp(min=0)
        bloom_radius = max(2, min(height, width) // 120)
        worked = worked + _gaussian_blur(highlights, bloom_radius) * values["bloom"]

        enhanced = source * values["preserve"] + worked * (1.0 - values["preserve"])
        mask = _mask_tensor(effect_mask, batch, height, width, device, dtype)
        if mask is not None:
            enhanced = source * (1.0 - mask) + enhanced * mask

        lighting_pass = (
            0.18
            + diffuse * 0.42
            + rim_map * values["rim"] * 0.18
            + specular_map * values["spec"] * 0.22
            - ao_map * values["ao"] * 0.20
        ).clamp(0, 1).repeat(1, 3, 1, 1)
        normal_preview = normals * 0.5 + 0.5
        depth_preview = depth_map.repeat(1, 3, 1, 1)
        return (
            _to_bhwc(enhanced),
            _to_bhwc(lighting_pass),
            _to_bhwc(normal_preview),
            _to_bhwc(depth_preview),
        )


def _nearest_color_name(value):
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(character * 2 for character in text)
    try:
        rgb = tuple(int(text[index:index + 2], 16) for index in (0, 2, 4))
    except (ValueError, IndexError):
        return "warm golden"
    colors = {
        "warm golden": (255, 205, 145),
        "warm white": (255, 238, 214),
        "neutral white": (235, 240, 245),
        "cool blue": (125, 190, 255),
        "moonlight blue": (105, 135, 220),
        "cyan": (75, 235, 245),
        "emerald green": (70, 220, 145),
        "magenta": (240, 85, 210),
        "rose pink": (255, 125, 165),
        "orange": (255, 145, 60),
        "deep red": (220, 55, 55),
        "violet": (145, 95, 245),
    }
    return min(
        colors,
        key=lambda name: sum((rgb[channel] - colors[name][channel]) ** 2 for channel in range(3)),
    )


class ICLightPromptBuilder:
    DIRECTIONS = [
        "left", "right", "top", "bottom", "front", "back_rim",
        "top_left", "top_right", "window_left", "window_right",
    ]
    STYLES = [
        "natural", "anime_luxury", "cinematic", "soft_studio",
        "dramatic", "neon", "jewel_glow", "volumetric", "moonlight",
    ]
    TIMES = ["auto", "daylight", "golden_hour", "sunset", "night", "indoor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "direction": (cls.DIRECTIONS, {"default": "top_left"}),
                "style": (cls.STYLES, {"default": "anime_luxury"}),
                "time_of_day": (cls.TIMES, {"default": "auto"}),
                "light_color": ("STRING", {"default": "#ffd9ad"}),
                "intensity": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "preserve_character": ("BOOLEAN", {"default": True}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "build"
    CATEGORY = "RTX Illustration/IC-Light"
    DESCRIPTION = "Builds IC-Light-friendly positive and negative relighting prompts."

    def build(
        self, direction, style, time_of_day, light_color,
        intensity, preserve_character, custom_prompt,
    ):
        direction_prompts = {
            "left": "key light coming from the left",
            "right": "key light coming from the right",
            "top": "overhead key light",
            "bottom": "dramatic light coming from below",
            "front": "soft frontal beauty light",
            "back_rim": "strong backlight with elegant rim lighting",
            "top_left": "key light coming from the upper left",
            "top_right": "key light coming from the upper right",
            "window_left": "natural window light entering from the left",
            "window_right": "natural window light entering from the right",
        }
        style_prompts = {
            "natural": "natural physically plausible illumination, gentle tonal transitions",
            "anime_luxury": "luxurious anime illustration lighting, polished highlights, rich dimensional shading",
            "cinematic": "cinematic lighting, deep controlled shadows, filmic contrast",
            "soft_studio": "large softbox studio lighting, soft shadows, clean skin tones",
            "dramatic": "dramatic high-contrast illumination, sculpted shadows",
            "neon": "colorful neon lighting, luminous edge accents, cyberpunk atmosphere",
            "jewel_glow": "radiant jewel-like highlights, magical sparkle, elegant glow",
            "volumetric": "volumetric light rays, atmospheric light scattering, visible light beams",
            "moonlight": "soft moonlight, cool nocturnal illumination, delicate rim light",
        }
        time_prompts = {
            "auto": "",
            "daylight": "clear daylight",
            "golden_hour": "warm golden-hour sunlight",
            "sunset": "rich sunset illumination",
            "night": "night scene illumination",
            "indoor": "controlled indoor illumination",
        }
        if intensity < 0.34:
            intensity_prompt = "subtle low-intensity"
        elif intensity < 0.72:
            intensity_prompt = "balanced medium-intensity"
        else:
            intensity_prompt = "powerful high-intensity"
        color_name = _nearest_color_name(light_color)
        parts = [
            direction_prompts[direction],
            style_prompts[style],
            time_prompts[time_of_day],
            f"{intensity_prompt} {color_name} light",
            "consistent light direction, coherent cast shadows, detailed highlights",
        ]
        if preserve_character:
            parts.append(
                "preserve the character identity, facial features, hairstyle, outfit design, "
                "line art and original composition"
            )
        if custom_prompt.strip():
            parts.append(custom_prompt.strip())
        positive = ", ".join(part for part in parts if part)

        negatives = [
            "inconsistent lighting", "multiple conflicting shadows", "flat lighting",
            "burned highlights", "crushed blacks", "color banding", "halo artifacts",
        ]
        if preserve_character:
            negatives.extend([
                "changed identity", "different face", "changed hairstyle",
                "changed clothes", "deformed hands", "extra fingers", "distorted line art",
            ])
        return (positive, ", ".join(negatives))


class ICLightDetailFinish:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "ic_light_image": ("IMAGE",),
                "relight_strength": ("FLOAT", {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_recovery": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.5, "step": 0.01}),
                "color_preservation": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "highlight_protection": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "effect_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("finished", "relight_only")
    FUNCTION = "finish"
    CATEGORY = "RTX Illustration/IC-Light"
    DESCRIPTION = "Restores original color and high-frequency illustration details after IC-Light."

    def finish(
        self, original, ic_light_image, relight_strength,
        detail_recovery, color_preservation, highlight_protection,
        effect_mask=None,
    ):
        source = _to_bchw(original).clamp(0, 1)
        relit = _to_bchw(ic_light_image).to(device=source.device, dtype=source.dtype)
        batch, _, height, width = source.shape
        relit = F.interpolate(relit, size=(height, width), mode="bicubic", align_corners=False)
        if relit.shape[0] == 1 and batch > 1:
            relit = relit.expand(batch, -1, -1, -1)
        elif source.shape[0] == 1 and relit.shape[0] > 1:
            source = source.expand(relit.shape[0], -1, -1, -1)
            batch = relit.shape[0]
        else:
            common_batch = min(batch, relit.shape[0])
            source, relit, batch = source[:common_batch], relit[:common_batch], common_batch

        relit = relit.clamp(0, 1)
        source_luma = (
            0.2126 * source[:, 0:1]
            + 0.7152 * source[:, 1:2]
            + 0.0722 * source[:, 2:3]
        )
        relit_luma = (
            0.2126 * relit[:, 0:1]
            + 0.7152 * relit[:, 1:2]
            + 0.0722 * relit[:, 2:3]
        )
        source_chroma = source - source_luma
        relit_chroma = relit - relit_luma
        preserved_color = relit_luma + (
            relit_chroma * (1.0 - color_preservation)
            + source_chroma * color_preservation
        )

        radius = max(1, min(height, width) // 512)
        original_detail = source - _gaussian_blur(source, radius)
        worked = preserved_color + original_detail * detail_recovery

        if highlight_protection > 0:
            highlight_mask = ((worked - 0.72) / 0.28).clamp(0, 1)
            compressed = 0.72 + (worked - 0.72).clamp(min=0) / (
                1.0 + highlight_protection * 2.5 * (worked - 0.72).clamp(min=0)
            )
            worked = worked * (1.0 - highlight_mask * highlight_protection) + compressed * (
                highlight_mask * highlight_protection
            )

        finished = source * (1.0 - relight_strength) + worked * relight_strength
        mask = _mask_tensor(effect_mask, batch, height, width, source.device, source.dtype)
        if mask is not None:
            finished = source * (1.0 - mask) + finished * mask
        return (_to_bhwc(finished), _to_bhwc(relit))


NODE_CLASS_MAPPINGS = {
    "RTXIllustrationEnhancer": RTXIllustrationEnhancer,
    "ICLightPromptBuilder": ICLightPromptBuilder,
    "ICLightDetailFinish": ICLightDetailFinish,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RTXIllustrationEnhancer": "RTX Illustration Enhancer ✨",
    "ICLightPromptBuilder": "IC-Light Prompt Builder 💡",
    "ICLightDetailFinish": "IC-Light Detail Finish ✨",
}
