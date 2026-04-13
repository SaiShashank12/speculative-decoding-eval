"""
EAGLE Draft Head Training for Med-Gemma 4B
==========================================
Simplified, single-device (MPS/CPU) training script.
Trains a lightweight EAGLE-style draft head on top of frozen Med-Gemma 4B.

The draft head learns to predict the next hidden state from the current
hidden state + token embedding, enabling speculative decoding.

Usage:
    python train_eagle_medgemma.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import os
import json
import time
import math

# ============================================================================
# Configuration
# ============================================================================

BASE_MODEL_ID = "google/medgemma-4b-it"
DEVICE = "cpu"  # Running on CPU for guaranteed fp32 numerical stability
DTYPE = torch.float32

# Med-Gemma 4B text config
HIDDEN_SIZE = 2560
INTERMEDIATE_SIZE = 10240
NUM_ATTENTION_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = 256
RMS_NORM_EPS = 1e-6
VOCAB_SIZE = 262208

# Training config
MAX_SEQ_LEN = 256       # Short sequences to fit in memory
BATCH_SIZE = 1           # Small batch for memory
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5     # Reduced LR to prevent early divergence
LOOKAHEAD = 3            # Predict up to 3 steps ahead
SAVE_DIR = "eagle_medgemma_weights"

# ============================================================================
# Draft Head Architecture
# ============================================================================

class GemmaRMSNorm(nn.Module):
    """Gemma-style RMS normalization (adds 1 to weight as per Gemma convention)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        input_dtype = x.dtype
        # Numerical stability fix
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return ((self.weight + 1.0) * x_normed).to(input_dtype)

class RotaryEmbedding(nn.Module):
    """Simplified RoPE for the draft head."""
    def __init__(self, dim, max_position_embeddings=8192, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_position_embeddings

    def forward(self, x, position_ids):
        # x: [batch, heads, seq_len, head_dim]
        # Always compute freqs in fp32
        inv_freq = self.inv_freq.to(torch.float32).to(x.device)
        freqs = torch.einsum("i,j->ij", position_ids[0].float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]  # [1, 1, seq_len, dim]
        sin = emb.sin()[None, None, :, :]
        return cos.to(x.dtype), sin.to(x.dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class DraftAttention(nn.Module):
    """Simplified self-attention for the draft head."""
    def __init__(self):
        super().__init__()
        self.num_heads = NUM_ATTENTION_HEADS
        self.num_kv_heads = NUM_KV_HEADS
        self.head_dim = HEAD_DIM
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        input_dim = HIDDEN_SIZE * 2  # concat(hidden, emb)

        self.q_proj = nn.Linear(input_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, HIDDEN_SIZE, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(q, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_groups > 1:
            k = k[:, :, None, :, :].expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(bsz, self.num_heads, q_len, self.head_dim)
            v = v[:, :, None, :, :].expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(bsz, self.num_heads, q_len, self.head_dim)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax in fp32 for stability
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)

        return self.o_proj(attn_output)

class DraftMLP(nn.Module):
    """Gemma-style gated MLP for the draft head."""
    def __init__(self):
        super().__init__()
        draft_intermediate = INTERMEDIATE_SIZE // 4
        self.gate_proj = nn.Linear(HIDDEN_SIZE, draft_intermediate, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, draft_intermediate, bias=False)
        self.down_proj = nn.Linear(draft_intermediate, HIDDEN_SIZE, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

class DraftDecoderLayer(nn.Module):
    """Single transformer layer for the draft head."""
    def __init__(self):
        super().__init__()
        self.self_attn = DraftAttention()
        self.mlp = DraftMLP()
        self.input_layernorm = GemmaRMSNorm(HIDDEN_SIZE * 2, eps=RMS_NORM_EPS)
        self.post_attention_layernorm = GemmaRMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS)

    def forward(self, hidden_states, emb, attention_mask=None, position_ids=None):
        combined = torch.cat((hidden_states, emb), dim=-1)
        combined = self.input_layernorm(combined)

        attn_output = self.self_attn(combined, attention_mask, position_ids)
        hidden_states = hidden_states + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class EAGLEDraftHead(nn.Module):
    """EAGLE-style draft head for Med-Gemma 4B."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.layer = DraftDecoderLayer()
        self.norm = GemmaRMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS)

    def forward(self, hidden_states, token_embeds, attention_mask=None, position_ids=None):
        h = self.fc(hidden_states)
        h = self.layer(h, token_embeds, attention_mask, position_ids)
        h = self.norm(h)
        return h

# ============================================================================
# Dataset
# ============================================================================

class MedicalTextDataset(Dataset):
    """Simple dataset for training - uses medical text prompts."""

    MEDICAL_PROMPTS = [
        "A 45-year-old male presents with acute substernal chest pain radiating to the left arm. The pain started 2 hours ago while at rest. He has a history of hypertension and smoking. ECG shows ST-elevation in leads II, III, and aVF. The most likely diagnosis is inferior myocardial infarction.",
        "Differential diagnosis for a patient with persistent cough lasting more than 3 weeks includes tuberculosis, lung cancer, chronic obstructive pulmonary disease, gastroesophageal reflux disease, and postnasal drip.",
        "Type 2 diabetes mellitus is characterized by insulin resistance and relative insulin deficiency. First-line pharmacological treatment typically involves metformin, which works primarily by decreasing hepatic glucose production.",
        "The Glasgow Coma Scale assesses eye opening response, verbal response, and motor response. A score of 3 indicates deep coma while 15 indicates fully alert. Scores below 8 generally indicate severe brain injury.",
        "Hypertension is defined as a systolic blood pressure of 130 mmHg or higher, or a diastolic blood pressure of 80 mmHg or higher. First-line treatment includes lifestyle modifications such as dietary changes, exercise, and weight management.",
        "Pneumonia presents with fever, productive cough, pleuritic chest pain, and dyspnea. Community-acquired pneumonia is most commonly caused by Streptococcus pneumoniae. Chest X-ray findings include consolidation and air bronchograms.",
        "Acute appendicitis typically presents with periumbilical pain that migrates to the right lower quadrant. McBurney's point tenderness is a classic finding. Treatment is appendectomy, which can be performed laparoscopically.",
        "Congestive heart failure is categorized as systolic or diastolic dysfunction. Symptoms include dyspnea, orthopnea, paroxysmal nocturnal dyspnea, and peripheral edema. Treatment includes ACE inhibitors, beta-blockers, and diuretics.",
        "Chronic kidney disease is staged based on glomerular filtration rate. Stage 1 has a GFR of 90 or above with evidence of kidney damage. Stage 5, also known as end-stage renal disease, has a GFR below 15.",
        "Stroke can be ischemic or hemorrhagic. The FAST mnemonic helps identify stroke symptoms: Face drooping, Arm weakness, Speech difficulty, Time to call emergency services. Thrombolytic therapy with alteplase may be given within 4.5 hours.",
        "Asthma is a chronic inflammatory airway disease characterized by reversible airflow obstruction, bronchial hyperresponsiveness, and airway inflammation. Management follows a stepwise approach with inhaled corticosteroids as the cornerstone.",
        "Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection. The qSOFA criteria include altered mental status, systolic blood pressure of 100 mmHg or lower, and respiratory rate of 22 or higher.",
        "Deep vein thrombosis risk factors include immobilization, surgery, malignancy, and oral contraceptives. Diagnosis involves D-dimer testing and compression ultrasonography. Treatment includes anticoagulation with heparin followed by warfarin.",
        "Cirrhosis is the end stage of chronic liver disease. Complications include ascites, variceal bleeding, hepatic encephalopathy, and hepatocellular carcinoma. The Child-Pugh score classifies severity into classes A, B, and C.",
        "Rheumatoid arthritis is a systemic autoimmune disease characterized by symmetric polyarthritis affecting small joints. Rheumatoid factor and anti-CCP antibodies are important serological markers. Disease-modifying agents including methotrexate are first-line.",
        "Thyroid function tests include TSH, free T4, and free T3. Hypothyroidism presents with fatigue, weight gain, cold intolerance, and constipation. Treatment involves levothyroxine replacement therapy with dose titration based on TSH levels.",
    ]

    def __init__(self, tokenizer, max_len=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        for prompt in self.MEDICAL_PROMPTS:
            tokens = tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=max_len, padding="max_length")
            self.data.append({
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================================
# Training Loop
# ============================================================================

def make_causal_mask(seq_len, dtype, device):
    """Create a causal attention mask."""
    # Use -10000.0 instead of finfo.min to prevent softmax NaN on Mac/MPS
    mask = torch.triu(torch.full((seq_len, seq_len), -10000.0, device=device), diagonal=1)
    return mask[None, None, :, :]


def train():
    print(f"Device: {DEVICE}")
    print(f"Loading base model: {BASE_MODEL_ID}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load frozen base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        low_cpu_mem_usage=True,
        output_hidden_states=True,
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    print("Base model loaded!")

    # Get embedding layer and LM head references
    # Gemma3ForConditionalGeneration structure:
    #   model.language_model.embed_tokens  (Gemma3TextScaledWordEmbedding)
    #   lm_head (Linear)
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'language_model'):
        embed_tokens = base_model.model.language_model.embed_tokens
        lm_head = base_model.lm_head
    elif hasattr(base_model, 'language_model'):
        embed_tokens = base_model.language_model.embed_tokens
        lm_head = base_model.lm_head
    elif hasattr(base_model, 'model') and hasattr(base_model.model, 'embed_tokens'):
        embed_tokens = base_model.model.embed_tokens
        lm_head = base_model.lm_head
    else:
        raise RuntimeError("Cannot find embedding layer!")

    # Create draft head
    draft_head = EAGLEDraftHead().to(DEVICE).to(torch.float32)
    num_params = sum(p.numel() for p in draft_head.parameters() if p.requires_grad)
    print(f"Draft head parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # Dataset
    dataset = MedicalTextDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset size: {len(dataset)} samples")

    # Optimizer
    optimizer = torch.optim.AdamW(draft_head.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS * len(dataloader)
    )
    criterion = nn.SmoothL1Loss()

    # Training
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting EAGLE draft head training")
    print(f"Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"Lookahead: {LOOKAHEAD} steps")
    print(f"{'='*60}\n")

    for epoch in range(NUM_EPOCHS):
        draft_head.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            # --- Step 1: Get frozen base model hidden states ---
            with torch.no_grad():
                outputs = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                # Get last hidden state as target
                hidden_states = outputs.hidden_states[-1]  # [B, S, H]
                logits_target = outputs.logits              # [B, S, V] - target logits

                # Get token embeddings
                token_embeds = embed_tokens(input_ids).detach()  # [B, S, H]

            # --- Step 2: Train draft head ---
            # For each position t, predict hidden state at t+1
            # Input: hidden_states[:, :-1], token_embeds[:, :-1]
            # Target: hidden_states[:, 1:]
            seq_len = hidden_states.shape[1] - 1
            if seq_len <= 0:
                continue

            h_input = hidden_states[:, :-1].float()   # [B, S-1, H]
            e_input = token_embeds[:, :-1].float()     # [B, S-1, H]
            h_target = hidden_states[:, 1:].float()    # [B, S-1, H]

            position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
            causal_mask = make_causal_mask(seq_len, torch.float32, DEVICE)

            # Forward through draft head
            h_pred = draft_head(h_input, e_input, causal_mask, position_ids)

            # --- Step 3: Compute loss ---
            # Primary loss: predict the next hidden state (SmoothL1)
            loss = criterion(h_pred, h_target.detach())

            # --- Step 4: Compute accuracy (top-1 token match) ---
            with torch.no_grad():
                # Pass predicted hidden state through frozen LM head
                pred_logits = lm_head(h_pred.to(lm_head.weight.dtype))
                pred_tokens = pred_logits.argmax(dim=-1)     # [B, S-1]
                target_tokens = input_ids[:, 1:]              # [B, S-1]
                # Only count non-padding positions
                mask = attention_mask[:, 1:].bool()
                correct = (pred_tokens == target_tokens) & mask
                acc = correct.sum().float() / mask.sum().float()

            # --- Step 5: Backward ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(draft_head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            num_batches += 1

            if (batch_idx + 1) % 4 == 0 or batch_idx == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}, Token Acc: {acc.item():.2%}")

            # Clear gradients and caches
            if DEVICE == "mps":
                torch.mps.empty_cache()

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = epoch_acc / max(num_batches, 1)

        print(f"\n  Epoch {epoch+1} Summary: Avg Loss: {avg_loss:.4f}, "
              f"Avg Token Acc: {avg_acc:.2%}, Time: {epoch_time:.1f}s\n")

        # Save checkpoint
        save_path = os.path.join(SAVE_DIR, f"draft_head_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": draft_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "accuracy": avg_acc,
            "config": {
                "hidden_size": HIDDEN_SIZE,
                "intermediate_size": INTERMEDIATE_SIZE,
                "num_attention_heads": NUM_ATTENTION_HEADS,
                "num_kv_heads": NUM_KV_HEADS,
                "head_dim": HEAD_DIM,
                "base_model": BASE_MODEL_ID,
            }
        }, save_path)
        print(f"  Saved checkpoint: {save_path}")

    # Save final config
    config_path = os.path.join(SAVE_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "base_model": BASE_MODEL_ID,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_attention_heads": NUM_ATTENTION_HEADS,
            "num_key_value_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "rms_norm_eps": RMS_NORM_EPS,
            "vocab_size": VOCAB_SIZE,
            "lookahead": LOOKAHEAD,
            "draft_intermediate_ratio": 0.25,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete! Weights saved to: {SAVE_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
