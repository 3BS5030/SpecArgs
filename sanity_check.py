import torch
import torch.nn as nn
import torch.optim as optim
from pipeline.data import run_data_pipeline
from pipeline.preprocessing import run_preprocessing_pipeline
from pipeline.model import run_model_pipeline
from pipeline.evaluation import run_evaluation_pipeline
from member_1 import predict_from_wav_model_las, TOKENIZER

def run_sanity_check():
    print("=== 🧪 LAS Sanity Check: Overfitting on 3 Samples ===")
    
    print("\n[Step 1] Loading and shrinking data...")
    raw_data_dict = run_data_pipeline()
    if isinstance(raw_data_dict, dict):
        all_items = raw_data_dict.get("train", []) + raw_data_dict.get("test", [])
        small_subset = all_items[:3]
    else:
        small_subset = raw_data_dict[:3]

    data_split = {
        "train": small_subset,
        "valid": small_subset,
        "test": small_subset
    }

    print("\n[Step 2] Preprocessing...")
    processed_data = run_preprocessing_pipeline(data_split, gui_augment={"policy": "None"})
    
    print("\n[Step 3] Building LAS Model...")
    nc = processed_data.get("num_classes")
    seq_len = processed_data.get("target_sequence_length")
    model = run_model_pipeline(num_classes=nc, target_sequence_length=seq_len)
    
    # === التحكم الكامل: الـ Loss والـ Optimizer الصح ===
    criterion = nn.CrossEntropyLoss(ignore_index=2, label_smoothing=0.1) # 2 هو رقم الـ PAD
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    print(f"\n[Step 4] Training on 3 samples for 100 epochs (Bypassing Pipeline)...")
    device = next(model.parameters()).device
    train_loader = processed_data["train_loader"]
    
    # === لوب التدريب المباشر (عشان محدش يتدخل في حساباتنا) ===
    model.train()
    for epoch in range(1, 101):
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            logits = model(xb, target_seq=yb)
            # دمج الأبعاد عشان الـ CrossEntropy 
            loss = criterion(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 5 == 0:
            print(f"[Sanity Check] Epoch {epoch}/100 Loss = {total_loss/len(train_loader):.4f}")
    # =========================================================

    print("\n[Step 5] Practical Test (Inference)...")
    # استخراج أول ملف تم التدريب عليه من الداتا المجهزة
    test_wav_path = "data/LibriSpeech/dev-clean/" # هيتجاب من الـ tuple لو متاح
    
    # بدل ما نكتب المسار بإيدينا، هنجيب الصوت من الداتا اللي اتدربنا عليها مباشرة
    model.eval()
    with torch.no_grad():
        sample_x, sample_y = next(iter(train_loader))
        sample_x = sample_x[0:1].to(device)
        logits = model(sample_x, target_seq=None, max_len=seq_len)
        
        pred_ids = logits.argmax(dim=-1)[0].cpu().tolist()
        if TOKENIZER.eos_id in pred_ids:
            pred_ids = pred_ids[:pred_ids.index(TOKENIZER.eos_id)]
            
        clean_preds = [p for p in pred_ids if p not in (TOKENIZER.sos_id, TOKENIZER.eos_id, TOKENIZER.pad_id)]
        predicted_text = TOKENIZER.decode(clean_preds)
        
        target_ids = sample_y[0].tolist()
        clean_targets = [p for p in target_ids if p not in (TOKENIZER.sos_id, TOKENIZER.eos_id, TOKENIZER.pad_id)]
        actual_text = TOKENIZER.decode(clean_targets)
        
        print("\n" + "="*40)
        print("🎯 TARGET (النص الأصلي):")
        print(actual_text.upper())
        print("-" * 40)
        print("🤖 PREDICTION (توقع الموديل):")
        print(predicted_text.upper())
        print("="*40)

if __name__ == "__main__":
    run_sanity_check()