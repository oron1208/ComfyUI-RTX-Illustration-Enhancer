# IC-Light integration

## Required external node

Install `kijai/ComfyUI-IC-Light` with ComfyUI-Manager and place an IC-Light
SD1.5 model under `ComfyUI/models/unet/IC-Light`.

This repository does not import or bundle IC-Light, so its core nodes remain
available when IC-Light is not installed.

## Recommended graph

1. Load and optionally upscale the original illustration.
2. Feed the original image into `VAE Encode`.
3. Use `IC-Light Prompt Builder` outputs with positive and negative
   `CLIP Text Encode` nodes.
4. Patch the SD1.5 model with `Load And Apply IC-Light`.
5. Connect conditioning, VAE and the foreground latent to
   `IC-Light Conditioning`.
6. Sample and decode the relit image.
7. Connect the original image and decoded result to `IC-Light Detail Finish`.
8. Connect `finished` to `RTX Illustration Enhancer` for bloom, specular
   highlights and the final detail pass.

## Starting values

- IC-Light denoise: `0.65` to `0.85`
- Detail Finish relight strength: `0.80`
- Detail recovery: `0.45`
- Color preservation: `0.35`
- Highlight protection: `0.25`

For faces and line art, connect a subject mask to `effect_mask` or increase
color preservation and detail recovery.

