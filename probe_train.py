
summed_differences = None
valid_pairs_processed = 0

print(f"Starting extraction loop for {len(prompts)} pairs...")

# Enforce no gradients to save VRAM
with torch.no_grad():
    for i in tqdm(range(len(prompts)), desc="Training Probe"):
        try:
            question = prompts[i]
            pos_ans = positive_targets[i]
            neg_ans = negative_targets[i]

            # 1. Format Messages
            msg_pos = [{"role": "user", "content": question}, {"role": "assistant", "content": pos_ans}]
            msg_neg = [{"role": "user", "content": question}, {"role": "assistant", "content": neg_ans}]

            # 2. Tokenize
            input_pos = tokenizer.apply_chat_template(msg_pos, return_tensors="pt", add_generation_prompt=False).to(device)
            input_neg = tokenizer.apply_chat_template(msg_neg, return_tensors="pt", add_generation_prompt=False).to(device)

            # 3. Forward Pass (Positive)
            out_pos = model(**input_pos, output_hidden_states=True)
            # Tuple of (Layers), each shape [1, seq_len, hidden_dim]. We want [:, -1, :]
            acts_pos = torch.stack([layer[0, -1, :] for layer in out_pos.hidden_states])

            # Flush VRAM before the negative pass
            del out_pos
            torch.cuda.empty_cache()

            # 4. Forward Pass (Negative)
            out_neg = model(**input_neg, output_hidden_states=True)
            acts_neg = torch.stack([layer[0, -1, :] for layer in out_neg.hidden_states])

            # Flush VRAM again
            del out_neg
            torch.cuda.empty_cache()

            # 5. Calculate Difference for this pair
            delta = acts_pos - acts_neg # Shape: [num_layers, hidden_dim]

            # Initialize accumulator on first successful pass
            if summed_differences is None:
                summed_differences = torch.zeros_like(delta)

            # 6. Accumulate
            summed_differences += delta
            valid_pairs_processed += 1

            # Clean up loop variables
            del input_pos, input_neg, acts_pos, acts_neg, delta

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[Warning] OOM on pair {i}. Skipping and clearing cache...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

# 7. Calculate Mean and Normalize
print(f"\nSuccessfully processed {valid_pairs_processed} out of {len(prompts)} pairs.")

# Average the differences
mean_differences = summed_differences / valid_pairs_processed

# Normalize the vector at each layer independently (L2 norm)
# This matches Nguyen et al.'s methodology to ensure cross-comparison validity
layer_norms = torch.norm(mean_differences, p=2, dim=1, keepdim=True)

# Add a tiny epsilon to prevent division by zero in case of a dead layer
eval_awareness_probes = mean_differences / (layer_norms + 1e-8)

num_layers, hidden_dim = eval_awareness_probes.shape
print(f"Extraction Complete!")
print(f"Resulting Probe Matrix Shape: {num_layers} layers x {hidden_dim} dimensions")
