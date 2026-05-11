"""
Unit test for topk_sigmoid kernel with num_token_non_padded parameter.
Tests both scenarios: num_token_non_padded is None and not None.
"""

import torch
import torch.nn.functional as F
import math

# Import the CUDA kernel - must import mcoplib.sgl_kernel first
import mcoplib.sgl_kernel


def reference_topk_sigmoid(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: torch.Tensor = None
) -> tuple:
    """
    Reference implementation of topk_sigmoid in PyTorch.

    Args:
        gating_output: [num_tokens, num_experts] - Router logits
        correction_bias: [num_experts] or None - Correction bias for each expert
        topk: Number of experts to select per token
        renormalize: Whether to renormalize the weights
        num_token_non_padded: Scalar tensor indicating number of valid tokens

    Returns:
        topk_weights: [num_tokens, topk] - Weights for selected experts
        topk_indices: [num_tokens, topk] - Indices of selected experts
    """
    # Convert to float32 for precision
    gating_fp32 = gating_output.float()
    bias_fp32 = correction_bias.float() if correction_bias is not None else None

    # 1. Apply sigmoid
    sigmoid_scores = torch.sigmoid(gating_fp32)

    # 2. Add bias for selection decision
    if bias_fp32 is not None:
        routing_scores = sigmoid_scores + bias_fp32
    else:
        routing_scores = sigmoid_scores

    # 3. Get top-k indices
    _, topk_indices = torch.topk(routing_scores, k=topk, dim=-1)

    # 4. Gather weights from original sigmoid scores (without bias)
    topk_weights = torch.gather(sigmoid_scores, dim=-1, index=topk_indices)

    # 5. Renormalize if required
    if renormalize:
        row_sum = topk_weights.sum(dim=-1, keepdim=True)
        row_sum = torch.where(row_sum > 0.0, row_sum, torch.ones_like(row_sum))
        topk_weights = topk_weights / row_sum

    # 6. Apply padding mask if num_token_non_padded is provided
    if num_token_non_padded is not None:
        num_valid = num_token_non_padded.item()
        num_tokens = topk_indices.shape[0]
        # Create mask for padded tokens
        indices = torch.arange(0, num_tokens, device=topk_indices.device)
        padding_mask = indices >= num_valid
        # Set indices to -1 for padded tokens
        topk_indices[padding_mask, :] = -1

    return topk_weights, topk_indices


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    cos_sim = F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0), dim=1)
    return cos_sim.item()


def test_topk_sigmoid_without_padding_no_param():
    """
    Test topk_sigmoid kernel WITHOUT passing num_token_non_padded parameter at all.
    This should match the original behavior and NOT raise an error.
    """
    print("\n" + "="*60)
    print("Test 1: topk_sigmoid WITHOUT num_token_non_padded parameter")
    print("="*60)

    # Configuration parameters
    NUM_ROUTED_EXPERTS = 256
    NUM_EXPERTS_PER_TOK = 8
    RENORMALIZE = True
    NUM_TOKENS = 128  # Use smaller batch for testing

    # Fixed random seed
    torch.manual_seed(42)

    # Create input tensors
    gating_output = torch.randn(
        (NUM_TOKENS, NUM_ROUTED_EXPERTS),
        dtype=torch.bfloat16,
        device='cuda'
    )
    correction_bias = torch.randn(
        NUM_ROUTED_EXPERTS,
        dtype=torch.float32,
        device='cuda'
    )

    # Allocate output tensors for CUDA kernel
    cuda_topk_weights = torch.empty(
        (NUM_TOKENS, NUM_EXPERTS_PER_TOK),
        dtype=torch.float32,
        device='cuda'
    )
    cuda_topk_indices = torch.empty(
        (NUM_TOKENS, NUM_EXPERTS_PER_TOK),
        dtype=torch.int32,
        device='cuda'
    )

    # Get reference results
    ref_weights, ref_indices = reference_topk_sigmoid(
        gating_output,
        correction_bias,
        NUM_EXPERTS_PER_TOK,
        RENORMALIZE,
        num_token_non_padded=None
    )

    # Run CUDA kernel - DO NOT pass num_token_non_padded at all
    # This tests that the parameter is truly optional
    torch.ops.sgl_kernel.topk_sigmoid(
        cuda_topk_weights,
        cuda_topk_indices,
        gating_output,
        RENORMALIZE,
        correction_bias
        # num_token_non_padded is NOT passed - should use default None
    )

    torch.cuda.synchronize()

    # Verify indices
    indices_match = torch.equal(ref_indices.to(torch.int32), cuda_topk_indices)
    print(f"Indices match: {indices_match}")

    if not indices_match:
        # Check if indices differ only by a small amount
        diff_indices = torch.abs(ref_indices.to(torch.int32) - cuda_topk_indices)
        max_diff = diff_indices.max().item()
        print(f"Max indices difference: {max_diff}")

    # Verify weights using cosine similarity
    cos_sim_weights = cosine_similarity(ref_weights, cuda_topk_weights)
    print(f"Weights cosine similarity: {cos_sim_weights:.8f}")

    max_diff_weights = torch.max(torch.abs(ref_weights - cuda_topk_weights)).item()
    print(f"Max weights difference: {max_diff_weights:.8e}")

    # Assert precision requirements
    assert cos_sim_weights > 0.9999, f"Cosine similarity {cos_sim_weights} < 0.9999"
    assert not math.isnan(cos_sim_weights), "Cosine similarity is NaN"

    print("Test 1 PASSED: Precision requirements met!")
    return True


def test_topk_sigmoid_without_padding_pass_none():
    """
    Test topk_sigmoid kernel passing num_token_non_padded=None explicitly.
    This should also work and match original behavior.
    """
    print("\n" + "="*60)
    print("Test 2: topk_sigmoid WITH num_token_non_padded=None explicitly")
    print("="*60)

    # Configuration parameters
    NUM_ROUTED_EXPERTS = 256
    NUM_EXPERTS_PER_TOK = 8
    RENORMALIZE = True
    NUM_TOKENS = 128

    # Fixed random seed
    torch.manual_seed(42)

    # Create input tensors
    gating_output = torch.randn(
        (NUM_TOKENS, NUM_ROUTED_EXPERTS),
        dtype=torch.bfloat16,
        device='cuda'
    )
    correction_bias = torch.randn(
        NUM_ROUTED_EXPERTS,
        dtype=torch.float32,
        device='cuda'
    )

    # Allocate output tensors for CUDA kernel
    cuda_topk_weights = torch.empty(
        (NUM_TOKENS, NUM_EXPERTS_PER_TOK),
        dtype=torch.float32,
        device='cuda'
    )
    cuda_topk_indices = torch.empty(
        (NUM_TOKENS, NUM_EXPERTS_PER_TOK),
        dtype=torch.int32,
        device='cuda'
    )

    # Get reference results
    ref_weights, ref_indices = reference_topk_sigmoid(
        gating_output,
        correction_bias,
        NUM_EXPERTS_PER_TOK,
        RENORMALIZE,
        num_token_non_padded=None
    )

    # Run CUDA kernel - pass None explicitly
    torch.ops.sgl_kernel.topk_sigmoid(
        cuda_topk_weights,
        cuda_topk_indices,
        gating_output,
        RENORMALIZE,
        correction_bias,
        None  # num_token_non_padded=None
    )

    torch.cuda.synchronize()

    # Verify indices
    indices_match = torch.equal(ref_indices.to(torch.int32), cuda_topk_indices)
    print(f"Indices match: {indices_match}")

    # Verify weights using cosine similarity
    cos_sim_weights = cosine_similarity(ref_weights, cuda_topk_weights)
    print(f"Weights cosine similarity: {cos_sim_weights:.8f}")

    # Assert precision requirements
    assert cos_sim_weights > 0.9999, f"Cosine similarity {cos_sim_weights} < 0.9999"
    assert not math.isnan(cos_sim_weights), "Cosine similarity is NaN"

    print("Test 2 PASSED: Precision requirements met!")
    return True


def test_topk_sigmoid_with_padding():
    """
    Test topk_sigmoid kernel WITH num_token_non_padded parameter.
    This tests the padding mask functionality.
    """
    print("\n" + "="*60)
    print("Test 3: topk_sigmoid WITH num_token_non_padded tensor")
    print("="*60)

    # Configuration parameters
    NUM_ROUTED_EXPERTS = 256
    NUM_EXPERTS_PER_TOK = 8
    RENORMALIZE = True
    NUM_TOKENS = 128  # Total tokens including padding
    NUM_VALID_TOKENS = 96  # Number of valid tokens (padding starts at 96)

    # Fixed random seed
    torch.manual_seed(42)

    # Create input tensors
    gating_output = torch.randn(
        (NUM_TOKENS, NUM_ROUTED_EXPERTS),
        dtype=torch.bfloat16,
        device='cuda'
    )
    correction_bias = torch.randn(
        NUM_ROUTED_EXPERTS,
        dtype=torch.float32,
        device='cuda'
    )

    # Create num_token_non_padded tensor
    num_token_non_padded = torch.tensor(NUM_VALID_TOKENS, dtype=torch.int64, device='cuda')

    # Allocate output tensors for CUDA kernel
    cuda_topk_weights = torch.empty(
        (NUM_TOKENS, NUM_EXPERTS_PER_TOK),
        dtype=torch.float32,
        device='cuda'
    )
    cuda_topk_indices = torch.empty(
        (NUM_TOKENS, NUM_EXPERTS_PER_TOK),
        dtype=torch.int32,
        device='cuda'
    )

    # Get reference results
    ref_weights, ref_indices = reference_topk_sigmoid(
        gating_output,
        correction_bias,
        NUM_EXPERTS_PER_TOK,
        RENORMALIZE,
        num_token_non_padded=num_token_non_padded
    )

    # Run CUDA kernel using torch.ops.sgl_kernel.topk_sigmoid
    torch.ops.sgl_kernel.topk_sigmoid(
        cuda_topk_weights,
        cuda_topk_indices,
        gating_output,
        RENORMALIZE,
        correction_bias,
        num_token_non_padded
    )

    torch.cuda.synchronize()

    # Verify indices for valid tokens (before padding)
    valid_indices_match = torch.equal(
        ref_indices[:NUM_VALID_TOKENS].to(torch.int32),
        cuda_topk_indices[:NUM_VALID_TOKENS]
    )
    print(f"Valid indices match: {valid_indices_match}")

    # Verify indices for padded tokens (should all be -1)
    padded_indices_cuda = cuda_topk_indices[NUM_VALID_TOKENS:]
    padded_indices_ref = ref_indices[NUM_VALID_TOKENS:]
    all_padded_are_minus_one = torch.all(padded_indices_cuda == -1).item()
    print(f"Padded indices are -1 (CUDA): {all_padded_are_minus_one}")

    padded_match = torch.equal(padded_indices_cuda, padded_indices_ref.to(torch.int32))
    print(f"Padded indices match reference: {padded_match}")

    # Verify weights for valid tokens using cosine similarity
    cos_sim_weights = cosine_similarity(ref_weights[:NUM_VALID_TOKENS], cuda_topk_weights[:NUM_VALID_TOKENS])
    print(f"Weights cosine similarity (valid tokens): {cos_sim_weights:.8f}")

    max_diff_weights = torch.max(torch.abs(ref_weights[:NUM_VALID_TOKENS] - cuda_topk_weights[:NUM_VALID_TOKENS])).item()
    print(f"Max weights difference (valid tokens): {max_diff_weights:.8e}")

    # Assert precision requirements
    assert valid_indices_match, "Valid indices do not match"
    assert all_padded_are_minus_one, "Padded indices are not -1"
    assert cos_sim_weights > 0.9999, f"Cosine similarity {cos_sim_weights} < 0.9999"
    assert not math.isnan(cos_sim_weights), "Cosine similarity is NaN"

    print("Test 3 PASSED: Precision requirements met!")
    return True


def test_topk_sigmoid_edge_cases():
    """
    Test edge cases for topk_sigmoid kernel with num_token_non_padded.
    """
    print("\n" + "="*60)
    print("Test 4: Edge Cases")
    print("="*60)

    NUM_ROUTED_EXPERTS = 256
    NUM_EXPERTS_PER_TOK = 8
    RENORMALIZE = True

    # Test case: num_token_non_padded = 0 (all tokens are padded)
    print("\n--- Subtest 4.1: All tokens padded ---")
    NUM_TOKENS = 64
    torch.manual_seed(42)

    gating_output = torch.randn((NUM_TOKENS, NUM_ROUTED_EXPERTS), dtype=torch.bfloat16, device='cuda')
    correction_bias = torch.randn(NUM_ROUTED_EXPERTS, dtype=torch.float32, device='cuda')
    num_token_non_padded = torch.tensor(0, dtype=torch.int64, device='cuda')

    cuda_topk_weights = torch.empty((NUM_TOKENS, NUM_EXPERTS_PER_TOK), dtype=torch.float32, device='cuda')
    cuda_topk_indices = torch.empty((NUM_TOKENS, NUM_EXPERTS_PER_TOK), dtype=torch.int32, device='cuda')

    torch.ops.sgl_kernel.topk_sigmoid(
        cuda_topk_weights,
        cuda_topk_indices,
        gating_output,
        RENORMALIZE,
        correction_bias,
        num_token_non_padded
    )
    torch.cuda.synchronize()

    all_minus_one = torch.all(cuda_topk_indices == -1).item()
    print(f"All indices are -1: {all_minus_one}")
    assert all_minus_one, "Indices should all be -1 when num_token_non_padded=0"
    print("Subtest 4.1 PASSED!")

    # Test case: num_token_non_padded = NUM_TOKENS (no padding)
    print("\n--- Subtest 4.2: No padding (num_token_non_padded = NUM_TOKENS) ---")
    NUM_TOKENS = 64
    torch.manual_seed(42)

    gating_output = torch.randn((NUM_TOKENS, NUM_ROUTED_EXPERTS), dtype=torch.bfloat16, device='cuda')
    correction_bias = torch.randn(NUM_ROUTED_EXPERTS, dtype=torch.float32, device='cuda')
    num_token_non_padded = torch.tensor(NUM_TOKENS, dtype=torch.int64, device='cuda')

    cuda_topk_weights = torch.empty((NUM_TOKENS, NUM_EXPERTS_PER_TOK), dtype=torch.float32, device='cuda')
    cuda_topk_indices = torch.empty((NUM_TOKENS, NUM_EXPERTS_PER_TOK), dtype=torch.int32, device='cuda')

    # Also run without num_token_non_padded for comparison (do not pass the parameter)
    cuda_topk_weights_ref = torch.empty((NUM_TOKENS, NUM_EXPERTS_PER_TOK), dtype=torch.float32, device='cuda')
    cuda_topk_indices_ref = torch.empty((NUM_TOKENS, NUM_EXPERTS_PER_TOK), dtype=torch.int32, device='cuda')

    torch.ops.sgl_kernel.topk_sigmoid(
        cuda_topk_weights,
        cuda_topk_indices,
        gating_output,
        RENORMALIZE,
        correction_bias,
        num_token_non_padded
    )

    # Do NOT pass num_token_non_padded here - should use default None
    torch.ops.sgl_kernel.topk_sigmoid(
        cuda_topk_weights_ref,
        cuda_topk_indices_ref,
        gating_output,
        RENORMALIZE,
        correction_bias
    )
    torch.cuda.synchronize()

    indices_match = torch.equal(cuda_topk_indices, cuda_topk_indices_ref)
    weights_match = torch.allclose(cuda_topk_weights, cuda_topk_weights_ref, rtol=1e-5, atol=1e-5)
    print(f"Indices match between with/without padding param: {indices_match}")
    print(f"Weights match between with/without padding param: {weights_match}")
    assert indices_match, "Results should be identical when num_token_non_padded = NUM_TOKENS"
    assert weights_match, "Weights should be identical when no padding"
    print("Subtest 4.2 PASSED!")

    # Test case: num_token_non_padded = 1 (only first token valid)
    print("\n--- Subtest 4.3: Only one valid token ---")
    NUM_TOKENS = 64
    torch.manual_seed(42)

    gating_output = torch.randn((NUM_TOKENS, NUM_ROUTED_EXPERTS), dtype=torch.bfloat16, device='cuda')
    correction_bias = torch.randn(NUM_ROUTED_EXPERTS, dtype=torch.float32, device='cuda')
    num_token_non_padded = torch.tensor(1, dtype=torch.int64, device='cuda')

    cuda_topk_weights = torch.empty((NUM_TOKENS, NUM_EXPERTS_PER_TOK), dtype=torch.float32, device='cuda')
    cuda_topk_indices = torch.empty((NUM_TOKENS, NUM_EXPERTS_PER_TOK), dtype=torch.int32, device='cuda')

    torch.ops.sgl_kernel.topk_sigmoid(
        cuda_topk_weights,
        cuda_topk_indices,
        gating_output,
        RENORMALIZE,
        correction_bias,
        num_token_non_padded
    )
    torch.cuda.synchronize()

    first_token_valid = cuda_topk_indices[0].min().item() >= 0
    rest_padded = torch.all(cuda_topk_indices[1:] == -1).item()
    print(f"First token has valid indices: {first_token_valid}")
    print(f"Rest of tokens are padded (-1): {rest_padded}")
    assert first_token_valid, "First token should have valid indices"
    assert rest_padded, "Rest of tokens should be padded"
    print("Subtest 4.3 PASSED!")

    print("\nTest 4 PASSED: All edge cases handled correctly!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running topk_sigmoid unit tests")
    print("="*60)

    results = []

    try:
        results.append(("Test 1: Without param (no num_token_non_padded)", test_topk_sigmoid_without_padding_no_param()))
    except Exception as e:
        print(f"Test 1 FAILED: {e}")
        results.append(("Test 1: Without param", False))

    try:
        results.append(("Test 2: With explicit None", test_topk_sigmoid_without_padding_pass_none()))
    except Exception as e:
        print(f"Test 2 FAILED: {e}")
        results.append(("Test 2: With explicit None", False))

    try:
        results.append(("Test 3: With padding tensor", test_topk_sigmoid_with_padding()))
    except Exception as e:
        print(f"Test 3 FAILED: {e}")
        results.append(("Test 3: With padding tensor", False))

    try:
        results.append(("Test 4: Edge cases", test_topk_sigmoid_edge_cases()))
    except Exception as e:
        print(f"Test 4 FAILED: {e}")
        results.append(("Test 4: Edge cases", False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")

    return all_passed


if __name__ == "__main__":
    run_all_tests()