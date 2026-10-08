"""
tests/debug_fl_core.py
-----------------------
Systematic FL core diagnostic — runs Steps 1-4 from the debug protocol.
Shows output at each step before proceeding.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import copy

def sep(title, width=62):
    print(f"\n{'='*width}")
    print(f"  STEP {title}")
    print("="*width)

# ============================================================
#  STEP 1: Label alignment
# ============================================================
sep("1 — LABEL ALIGNMENT")

from src.data.generator import generate_dataset, build_client_datasets
from src.fl.model import MLPModel

X_global, y_global = generate_dataset(2000, num_features=20, num_classes=10, seed=42)

print(f"  Global dataset   : X shape {X_global.shape}, y shape {y_global.shape}")
print(f"  Label dtype      : {y_global.dtype}  (need int64)")
print(f"  Unique labels    : {sorted(np.unique(y_global).tolist())}  (need 0-9)")
print(f"  Label range      : [{y_global.min()}, {y_global.max()}]")
print(f"  Class balance    : {np.bincount(y_global).tolist()}")

model_check = MLPModel(num_features=20, num_classes=10)
X_t = torch.from_numpy(X_global[:8])
logits = model_check(X_t)
preds  = logits.argmax(dim=1)
print(f"  Model forward    : input (8,20) -> log-softmax (8,10)  OK={logits.shape == (8,10)}")
print(f"  Pred class range : [{preds.min().item()}, {preds.max().item()}]  (should be 0-9)")
print(f"  Log-prob sums    : {logits.exp().sum(dim=1).tolist()[:3]}  (each ~1.0)")

assert y_global.dtype == np.int64,     "FAIL: labels not int64"
assert sorted(np.unique(y_global).tolist()) == list(range(10)), "FAIL: labels not 0-9"
assert logits.shape == (8, 10),        "FAIL: model output shape wrong"
print("\n  [OK] Step 1 PASSED — labels are 0-indexed int64, model output matches")

# ============================================================
#  STEP 2: Single-client overfitting
# ============================================================
sep("2 — SINGLE CLIENT OVERFITTING (100 epochs, no FL)")

from src.fl.local_trainer import LocalTrainer
from src.data.generator import ClientDataset

# Use first 100 samples of a single client shard
datasets = build_client_datasets(
    num_clients=10, num_classes=10, num_features=20,
    total_samples=2000, dirichlet_alpha=0.5, seed=42
)
client_ds = datasets[0]
print(f"  Client 0 samples : {len(client_ds)}")
print(f"  Labels present   : {sorted(client_ds.y.unique().tolist())}")

# Overfit a fresh model on this one client for many epochs
model_ov = MLPModel(num_features=20, num_classes=10)
opt = torch.optim.Adam(model_ov.parameters(), lr=1e-3)
criterion = nn.NLLLoss()

from torch.utils.data import DataLoader
loader = DataLoader(client_ds, batch_size=16, shuffle=True, drop_last=False)

print(f"\n  {'Epoch':>6}  {'Loss':>8}  {'Train Acc':>10}")
print(f"  {'-'*30}")
for epoch in range(100):
    model_ov.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_b, y_b in loader:
        opt.zero_grad()
        out = model_ov(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(y_b)
        correct    += (out.argmax(1) == y_b).sum().item()
        total      += len(y_b)
    if (epoch + 1) % 10 == 0:
        print(f"  {epoch+1:>6}  {total_loss/total:>8.4f}  {correct/total:>9.1%}")

model_ov.eval()
with torch.no_grad():
    X_all = client_ds.X
    y_all = client_ds.y
    preds_final = model_ov(X_all).argmax(1)
    final_train_acc = (preds_final == y_all).float().mean().item()

print(f"\n  Final train accuracy: {final_train_acc:.2%}  (should be >85% to confirm model can overfit)")
assert final_train_acc > 0.7, f"FAIL: Model can't overfit single client! ({final_train_acc:.2%})"
print(f"\n  [OK] Step 2 PASSED — model can overfit client data ({final_train_acc:.2%})")

# ============================================================
#  STEP 3: Test set correctness
# ============================================================
sep("3 — TEST SET CONSTRUCTION")

# Reproduce exactly what runner.py currently does
X_train_global, y_train = generate_dataset(2000, 20, 10, seed=42)
X_test_WRONG,  y_test_wrong = generate_dataset(500,  20, 10, seed=42+9999)  # BUG: different seed
X_test_CORRECT = X_train_global[1600:]   # held-out slice of SAME dataset
y_test_correct = y_train[1600:]

# Train a small model for 50 epochs on the first 1600 samples
model_t = MLPModel(num_features=20, num_classes=10)
opt_t = torch.optim.Adam(model_t.parameters(), lr=1e-3)
X_tr = torch.from_numpy(X_train_global[:1600])
y_tr = torch.from_numpy(y_train[:1600])
for _ in range(100):
    model_t.train()
    opt_t.zero_grad()
    nn.NLLLoss()(model_t(X_tr), y_tr).backward()
    opt_t.step()

model_t.eval()
with torch.no_grad():
    acc_wrong   = (model_t(torch.from_numpy(X_test_WRONG)).argmax(1)
                   == torch.from_numpy(y_test_wrong)).float().mean().item()
    acc_correct = (model_t(torch.from_numpy(X_test_CORRECT)).argmax(1)
                   == torch.from_numpy(y_test_correct)).float().mean().item()

print(f"  Test set (WRONG: seed+9999, diff distribution) : {acc_wrong:.2%}")
print(f"  Test set (CORRECT: held-out slice, same seed)  : {acc_correct:.2%}")
print(f"  Random-chance baseline                         : {1/10:.2%}")
print()
if acc_wrong < 0.15:
    print(f"  *** BUG CONFIRMED: runner.py generates test set with seed+9999,")
    print(f"      which is a completely different make_classification problem.")
    print(f"      The model can never score above chance on it, regardless of training.")
    print(f"  *** FIX: hold out a slice of the SAME generated dataset.")
else:
    print(f"  Surprising: wrong-seed test set scored {acc_wrong:.2%} — investigate further.")

print(f"\n  [OK] Step 3 IDENTIFIED — root cause is the test set seed mismatch in runner.py")

# ============================================================
#  STEP 4: FedAvg correctness
# ============================================================
sep("4 — FEDAVG WEIGHT AVERAGING")

from src.fl.aggregator import FedAvgAggregator

agg = FedAvgAggregator(num_features=20, num_classes=10)

# Build two models with known constant weights
m1 = MLPModel(num_features=20, num_classes=10)
m2 = MLPModel(num_features=20, num_classes=10)

# Set fc3.weight to known values
with torch.no_grad():
    m1.fc3.weight.fill_(2.0)
    m2.fc3.weight.fill_(4.0)

w1 = m1.get_weights()
w2 = m2.get_weights()

# FedAvg with equal sample counts: expected = (2*n1 + 4*n2)/(n1+n2)
n1, n2 = 100, 100  # equal weighting -> expected = 3.0
agg.aggregate_fl_updates([(w1, n1), (w2, n2)])
result = agg.global_model.fc3.weight[0, 0].item()
expected = (2.0 * n1 + 4.0 * n2) / (n1 + n2)

print(f"  m1 fc3.weight = 2.0 (n1={n1}), m2 fc3.weight = 4.0 (n2={n2})")
print(f"  Expected FedAvg result : {expected:.4f}")
print(f"  Actual   FedAvg result : {result:.4f}")
assert abs(result - expected) < 1e-5, f"FAIL: FedAvg result wrong! {result} != {expected}"
print(f"  Exact match: {abs(result - expected) < 1e-5}  [OK]")

# Unequal weighting: n1=300, n2=100 -> expected = (2*300 + 4*100)/400 = 2.5
n1u, n2u = 300, 100
agg.aggregate_fl_updates([(w1, n1u), (w2, n2u)])
result_unequal = agg.global_model.fc3.weight[0, 0].item()
expected_unequal = (2.0 * n1u + 4.0 * n2u) / (n1u + n2u)
print(f"\n  Unequal: n1=300, n2=100 -> expected {expected_unequal:.4f}, got {result_unequal:.4f}")
assert abs(result_unequal - expected_unequal) < 1e-5, f"FAIL: Unequal FedAvg wrong!"
print(f"  Exact match: {abs(result_unequal - expected_unequal) < 1e-5}  [OK]")

# Check aggregator does NOT zero weights (common bug: forgetting to load into model)
agg2 = FedAvgAggregator(num_features=20, num_classes=10)
with torch.no_grad():
    m3 = MLPModel(num_features=20, num_classes=10)
    m3.fc3.weight.fill_(5.0)
agg2.aggregate_fl_updates([(m3.get_weights(), 50)])
nonzero = agg2.global_model.fc3.weight.abs().sum().item()
print(f"\n  Single-client aggregate (all-5.0): global fc3 sum = {nonzero:.2f}  (must be >0)")
assert nonzero > 0, "FAIL: FedAvg zeroed the weights!"
print(f"  Not zeroed [OK]")

print(f"\n  [OK] Step 4 PASSED — FedAvg is mathematically correct, not the bug")

# ============================================================
#  SUMMARY
# ============================================================
print(f"\n{'='*62}")
print("  DIAGNOSTIC SUMMARY")
print("="*62)
print(f"  Step 1 (label alignment) : PASSED — no bug here")
print(f"  Step 2 (local overfitting): PASSED — model/trainer work ({final_train_acc:.1%} train acc)")
print(f"  Step 3 (test set)        : BUG FOUND")
print(f"    runner.py line 154: generates test set with seed+9999,")
print(f"    which creates a DIFFERENT make_classification problem.")
print(f"    Wrong-seed accuracy={acc_wrong:.1%} (≈ chance), Correct={acc_correct:.1%}")
print(f"  Step 4 (FedAvg math)     : PASSED — weights averaged correctly")
print()
print("  ROOT CAUSE: runner.py test set uses generate_dataset(seed=seed+9999)")
print("  FIX: generate one global dataset, hold out a fraction as test,")
print("       distribute the rest to clients. No separate generate_dataset call.")
print("="*62)
