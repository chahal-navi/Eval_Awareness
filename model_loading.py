
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing on: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


model_id = "microsoft/Phi-3-mini-4k-instruct"


print("\nLoading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)
# Standardizing the pad token for batching contrastive pairs later
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"\nLoading {model_id} into VRAM...")
print("Expect this to consume ~7.6 GB of your 15 GB limit.")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=False
)


model.eval()
for param in model.parameters():
    param.requires_grad = False

print("\nModel loaded successfully. Ready for activation extraction.")
if device == "cuda":
    print(f"Allocated VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
