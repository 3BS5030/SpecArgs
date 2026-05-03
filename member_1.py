import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from transformers import Wav2Vec2Model

# =========================
# CONFIGURATION
# =========================
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4 

# =========================
# TOKENIZER (Character-Level)
# =========================
class CharTokenizer:
    def __init__(self):
        # 0: SOS, 1: EOS, 2: PAD, 3: SPACE
        self.chars = ["<sos>", "<eos>", "<pad>", " "] + list("abcdefghijklmnopqrstuvwxyz'")
        self.char2id = {c: i for i, c in enumerate(self.chars)}
        self.id2char = {i: c for i, c in enumerate(self.chars)}
        self.sos_id = 0
        self.eos_id = 1
        self.pad_id = 2
        self.vocab_size = len(self.chars)

    def encode(self, text):
        text = text.lower()
        return [self.char2id[c] for c in text if c in self.char2id]

    def decode(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return "".join([self.id2char.get(i, "") for i in ids if i not in (self.sos_id, self.eos_id, self.pad_id)])

TOKENIZER = CharTokenizer()

# =========================
# DATA LOADING
# =========================
def load_raw_data(data_path="data", download_if_empty=True):
    dataset = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    abs_data = os.path.join(base_dir, data_path)
    os.makedirs(abs_data, exist_ok=True)
    
    try:
        print("[data] جاري تحميل بيانات LibriSpeech (حوالي 300MB).. قد يستغرق بضع دقائق لأول مرة.")
        ls = torchaudio.datasets.LIBRISPEECH(abs_data, url="dev-clean", download=download_if_empty)
        print(f"[data] تم تحميل داتا LibriSpeech: {len(ls)} ملف صوتي.")
        
        # subset_size = min(len(ls), 100) 
        for i in range(len(ls)):
            waveform, sr, transcript, _, _, _ = ls[i]
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
            
            encoded_text = TOKENIZER.encode(transcript)
            encoded_text.append(TOKENIZER.eos_id) 
            
            label = torch.tensor(encoded_text, dtype=torch.long)
            dataset.append((waveform, label))
            
    except Exception as ex:
        print(f"[data] فشل تحميل البيانات: {ex}")

    return dataset

# =========================
# HYBRID LAS MODEL ARCHITECTURE
# =========================
class HybridLASModel(nn.Module):
    def __init__(self, num_classes, target_sequence_length=200):
        super(HybridLASModel, self).__init__()
        self.sos_id = TOKENIZER.sos_id
        self.eos_id = TOKENIZER.eos_id
        self.pad_id = TOKENIZER.pad_id
        
        # 1. Encoder: Wav2Vec 2.0 (Pre-trained)
        print("[Hybrid] Loading Pre-trained Wav2Vec2 Encoder (facebook/wav2vec2-base-960h)...")
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        
        # تجميد أوزان الـ Encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # 2. Linear Bridge: لربط أبعاد Wav2Vec2 (768) بأبعاد الـ Decoder (512)
        self.bridge = nn.Linear(768, 512)
        
        # 3. Attention & Decoder (Custom)
        from member_3 import Attention
        from member_4 import Decoder
        
        self.attention = Attention(512, 512)
        self.decoder = Decoder(
            vocab_size=TOKENIZER.vocab_size,
            embed_dim=128, 
            hidden_dim=512, 
            encoder_dim=512
        )
        
        self.num_classes = num_classes
        self.target_sequence_length = target_sequence_length

    def beam_search_decode(self, x, beam_width=5, max_len=200):
        """
        Beam Search مخصص للـ Hybrid Architecture (Wav2Vec2 + LAS)
        """
        import torch.nn.functional as F
        device = x.device
        batch_size = x.size(0)
        assert batch_size == 1, "Beam Search works with batch_size=1"

        # 1. استخراج الخصائص من الـ Encoder الجاهز (Wav2Vec2)
        outputs = self.encoder(x)
        encoder_outputs = outputs.last_hidden_state
        encoder_outputs = self.bridge(encoder_outputs)

        # 2. تهيئة مسارات الـ Beam
        beam = [(0.0, [self.sos_id], None)]
        completed_hypotheses = []

        for t in range(max_len):
            new_beam = []
            for score, seq, hidden in beam:
                if seq[-1] == self.eos_id:
                    completed_hypotheses.append((score, seq))
                    continue

                dec_input = torch.tensor([[seq[-1]]], dtype=torch.long, device=device)

                if hidden is None:
                    query = torch.zeros(1, 512, device=device) # 512 هو حجم الـ Hidden state هنا
                else:
                    query = hidden[0][-1]

                context, _ = self.attention(query, encoder_outputs)
                logits, next_hidden = self.decoder(dec_input, hidden, context.unsqueeze(1))

                log_probs = F.log_softmax(logits.squeeze(1), dim=-1)
                topk_log_probs, topk_indices = torch.topk(log_probs[0], beam_width)

                for k in range(beam_width):
                    next_score = score + topk_log_probs[k].item()
                    next_seq = seq + [topk_indices[k].item()]
                    new_beam.append((next_score, next_seq, next_hidden))

            new_beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:beam_width]
            beam = [h for h in new_beam if h[1][-1] != self.eos_id]
            completed_hypotheses.extend([h for h in new_beam if h[1][-1] == self.eos_id])

            if len(beam) == 0:
                break

        if len(completed_hypotheses) == 0:
            completed_hypotheses = beam

        # معاقبة الجمل القصيرة جداً عشان نختار الجملة الأصح
        completed_hypotheses = sorted(completed_hypotheses, key=lambda x: x[0] / len(x[1]), reverse=True)
        return completed_hypotheses[0][1]
    
    def forward(self, x, target_seq=None, max_len=200):
        # x: [batch, time] - Raw Waveform
        # outputs = self.encoder(x)
        
        # ==== التعديل هنا: حماية كارت الشاشة ====
        self.encoder.eval() # وضع التقييم للموديل الجاهز
        with torch.no_grad(): # منع حساب الـ Gradients الثقيلة
            outputs = self.encoder(x)
            encoder_outputs = outputs.last_hidden_state
        encoder_outputs = self.bridge(encoder_outputs) # [batch, seq, 512]
        batch_size = x.size(0)
        device = x.device
        
        decoder_input = torch.full((batch_size, 1), self.sos_id, dtype=torch.long, device=device)
        decoder_hidden = None
        outputs = []
        
        seq_len = target_seq.size(1) if target_seq is not None else max_len
        use_teacher_forcing = True if target_seq is not None else False
        
        for t in range(seq_len):
            if decoder_hidden is None:
                query = torch.zeros(batch_size, 512, device=device)
            else:
                query = decoder_hidden[0][-1] 
                
            context, _ = self.attention(query, encoder_outputs)
            logits, decoder_hidden = self.decoder(decoder_input, decoder_hidden, context.unsqueeze(1))
            outputs.append(logits.unsqueeze(1))
            
            # Scheduled Teacher Forcing
            if use_teacher_forcing and random.random() < 0.75:
                decoder_input = target_seq[:, t].unsqueeze(1)
            else:
                decoder_input = logits.argmax(dim=-1).view(batch_size, 1).detach()
                
        return torch.cat(outputs, dim=1)

def build_embedding_layer(model=None, num_classes=None, use_lstm=True, target_sequence_length=None):
    nc = int(num_classes) if num_classes is not None else TOKENIZER.vocab_size
    las_model = HybridLASModel(
        num_classes=nc,
        target_sequence_length=target_sequence_length
    )
    return las_model.to(DEVICE)

# =========================
# TRAINING UTILS
# =========================
def initialize_optimizer(model, learning_rate=1e-3):
    if not isinstance(model, nn.Module):
        return model

    # تدريب البارامترات غير المجمدة فقط (الـ Decoder والـ Bridge)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    model.optimizer = optimizer
    model.criterion = nn.CrossEntropyLoss(ignore_index=TOKENIZER.pad_id, label_smoothing=0.1)
    model.train_device = DEVICE
    model.train_epochs = 100
    return model

# =========================
# INFERENCE & ACCURACY
# =========================
def load_wav_mono_16k_1d(path: str) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    waveform = waveform.squeeze() 
    waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var() + 1e-7)
    return waveform.unsqueeze(0) 

def predict_from_wav_model_las(model: nn.Module, wav_path: str) -> dict:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be nn.Module")

    x = load_wav_mono_16k_1d(wav_path).to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        max_length = getattr(model, "target_sequence_length", 200) or 200
        logits = model(x, target_seq=None, max_len=max_length)
        probs = torch.softmax(logits, dim=-1)[0].cpu()
        pred_ids = logits.argmax(dim=-1)[0].cpu().tolist()

    if TOKENIZER.eos_id in pred_ids:
        pred_ids = pred_ids[:pred_ids.index(TOKENIZER.eos_id)]
        
    clean_preds = [p for p in pred_ids if p not in (TOKENIZER.sos_id, TOKENIZER.eos_id, TOKENIZER.pad_id)]
    predicted_text = TOKENIZER.decode(clean_preds)

    basename = os.path.basename(wav_path)
    lines = [
        f"الموديل: Hybrid Wav2Vec2 + LAS",
        f"الملف: {basename}",
        "",
        "=== النص المستخرج (Transcript) ===",
        predicted_text.upper(),
        ""
    ]

    return {
        "pred": clean_preds,
        "probs": probs.tolist(),
        "display_text": "\n".join(lines),
        "n_classes": logits.shape[-1],
    }

def compute_accuracy(model):
    loader = getattr(model, "test_loader", None)
    if loader is None: return {"accuracy": 0.0}

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x, target_seq=y)
            preds = logits.argmax(dim=-1)
            mask = (y != TOKENIZER.pad_id)
            correct += ((preds == y) & mask).sum().item()
            total += mask.sum().item()

    acc = correct / total if total > 0 else 0.0
    model.last_accuracy = acc
    return {"accuracy": float(acc)}