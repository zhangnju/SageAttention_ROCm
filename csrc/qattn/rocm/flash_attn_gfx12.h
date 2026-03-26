#pragma once
#include <torch/extension.h>

torch::Tensor launch_flash_attn_gfx12(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool is_causal,
    float sm_scale);
