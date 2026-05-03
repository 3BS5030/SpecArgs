import torch
import difflib
import jiwer # مكتبة حساب الـ WER
from pipeline.data import run_data_pipeline
from pipeline.preprocessing import run_preprocessing_pipeline
from pipeline.model import run_model_pipeline
from member_1 import TOKENIZER

def evaluate_model_accuracy():
    print("=== 📊 Hybrid LAS Model Accuracy Evaluator (Greedy Search) ===")
    
    ckpt_path = "las_checkpoint.pth"
    
    print("\n[1] Loading Test Data...")
    raw_data = run_data_pipeline()
    processed_data = run_preprocessing_pipeline(raw_data, gui_augment={"policy": "None"})
    
    test_loader = processed_data.get("test_loader")
    if not test_loader:
        print("⚠️ لم يتم العثور على test_loader، سيتم استخدام valid_loader...")
        test_loader = processed_data.get("valid_loader")
        
    if not test_loader:
        print("❌ لا يوجد بيانات للاختبار!")
        return

    print("\n[2] Loading Model Checkpoint...")
    nc = processed_data.get("num_classes", getattr(TOKENIZER, "vocab_size", 31))
    seq_len = processed_data.get("target_sequence_length", 200)
    
    model = run_model_pipeline(num_classes=nc, target_sequence_length=seq_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print("✅ Checkpoint loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

    print("\n[3] Evaluating on Test Set (This may take a minute)...")
    
    total_chars = 0
    correct_chars = 0
    exact_sentences = 0
    total_sentences = 0
    
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(test_loader):
            xb, yb = xb.to(device), yb.to(device)
            
            # الـ Greedy Search السريع والمستقر
            logits = model(xb, target_seq=None, max_len=seq_len)
            preds = logits.argmax(dim=-1)
            
            for i in range(xb.size(0)):
                target_ids = yb[i].cpu().tolist()
                clean_targets = [p for p in target_ids if p not in (TOKENIZER.sos_id, TOKENIZER.eos_id, TOKENIZER.pad_id)]
                target_text = TOKENIZER.decode(clean_targets).strip()
                
                pred_ids = preds[i].cpu().tolist()
                if TOKENIZER.eos_id in pred_ids:
                    pred_ids = pred_ids[:pred_ids.index(TOKENIZER.eos_id)]
                clean_preds = [p for p in pred_ids if p not in (TOKENIZER.sos_id, TOKENIZER.eos_id, TOKENIZER.pad_id)]
                pred_text = TOKENIZER.decode(clean_preds).strip()
                
                if len(target_text) == 0: continue
                
                total_sentences += 1
                all_targets.append(target_text)
                all_preds.append(pred_text if len(pred_text) > 0 else " ")
                
                if target_text == pred_text:
                    exact_sentences += 1
                    
                matcher = difflib.SequenceMatcher(None, target_text, pred_text)
                ratio = matcher.ratio()
                
                total_chars += len(target_text)
                correct_chars += int(ratio * len(target_text))
                
                if total_sentences <= 5:
                    print(f"\n--- Example {total_sentences} ---")
                    print(f"🎯 Target: {target_text.upper()}")
                    print(f"🤖 Pred:   {pred_text.upper()}")
                    print(f"📈 Char Match: {ratio*100:.1f}%")

    print("\n" + "="*50)
    print("🏆 FINAL ACCURACY RESULTS 🏆")
    print("="*50)
    
    if total_sentences > 0:
        char_accuracy = (correct_chars / max(total_chars, 1)) * 100
        sentence_accuracy = (exact_sentences / max(total_sentences, 1)) * 100
        wer = jiwer.wer(all_targets, all_preds) * 100
        
        print(f"1️⃣ دقة الحروف (Character Accuracy): {char_accuracy:.2f}%")
        print(f"2️⃣ التطابق التام (Exact Sentence Match): {sentence_accuracy:.2f}%")
        print(f"3️⃣ معدل خطأ الكلمات (Word Error Rate - WER): {wer:.2f}% (كلما قل كان أفضل)")
        print(f"   (تم استخراج {exact_sentences} جملة بدون ولا غلطة من أصل {total_sentences} جملة)")
    else:
        print("⚠️ لم يتم العثور على جمل صالحة للتقييم.")
    print("="*50)

if __name__ == "__main__":
    evaluate_model_accuracy()