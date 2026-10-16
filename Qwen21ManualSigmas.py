import math

import torch


class Qwen21ManualSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 25, "min": 1, "max": 1000}),
                "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 16}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def get_sigmas(self, steps, width, height):
        # Official Qwen Image 2.1 FlowMatchEulerDiscreteScheduler dynamic shift.
        # mu depends on latent token count (spacial downscale ratio 16).
        tokens = max((height // 16) * (width // 16), 1)
        mu = 0.5 + (tokens - 256) * (0.9 - 0.5) / (8192 - 256)
        mu = min(max(mu, 0.0), 1.0)
        t = torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float64)
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / t - 1))
        # shift_terminal = 0.02: stretch so last non-zero sigma equals 0.02
        one_minus = 1 - sigmas
        sigmas = 1 - one_minus / (one_minus[-1] / (1 - 0.02))
        return (torch.cat([sigmas, sigmas.new_zeros(1)]).float(),)


NODE_CLASS_MAPPINGS = {
    "Qwen21ManualSigmas": Qwen21ManualSigmas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen21ManualSigmas": "Qwen2.1 Official Sigmas",
}
