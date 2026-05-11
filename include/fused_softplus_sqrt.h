#pragma once
#include <torch/extension.h>

torch::Tensor softplus_sqrt_cuda(torch::Tensor input);
