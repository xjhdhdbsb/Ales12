import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import collections
import re
import copy

# --- BPE токенизатор (остается на NumPy, так как это этап предобработки данных) ---
class BPETokenizer:
    def __init__(self, vocab_size=100):
        self.num_merges = vocab_size; self.vocab = []; self.merges = {}
    def _get_stats(self, word_freqs):
        pairs = collections.defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1): pairs[symbols[i], symbols[i+1]] += freq
        return pairs
    def _merge_vocab(self, pair, v_in):
        v_out = {}; bigram = re.escape(' '.join(pair)); p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in: v_out[p.sub(''.join(pair), word)] = v_in[word]
        return v_out
    def train(self, corpus):
        base_vocab_list = sorted(list(set("".join(corpus).replace(" ", ""))))
        word_freqs = collections.defaultdict(int)
        for text in corpus:
            for word in text.strip().split(): word_freqs[' '.join(list(word)) + ' </w>'] += 1
        for i in range(self.num_merges):
            pairs = self._get_stats(word_freqs)
            if not pairs: break
            best_pair = max(pairs, key=pairs.get)
            word_freqs = self._merge_vocab(best_pair, word_freqs); self.merges[best_pair] = i
        final_tokens = base_vocab_list + ["".join(token) if isinstance(token, tuple) else token for token in sorted(self.merges.keys(), key=self.merges.get)]
        self.special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.vocab = self.special_tokens + list(dict.fromkeys(final_tokens))
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        self.unk_id = self.token_to_id["<unk>"]
        print(f"BPE токенизатор обучен. Размер словаря: {len(self.vocab)}")
    def encode(self, text):
        pre_tokenized_words = [' '.join(list(word)) + ' </w>' for word in text.strip().split()]
        for pair, _ in sorted(self.merges.items(), key=lambda x: x[1]):
            for i, word in enumerate(pre_tokenized_words): pre_tokenized_words[i] = self._merge_vocab(pair, {word: 1}).popitem()[0]
        final_tokens = ' '.join(pre_tokenized_words).split()
        return [self.token_to_id.get(token, self.unk_id) for token in final_tokens]
    def decode(self, ids):
        tokens = [self.id_to_token.get(i, '<unk>') for i in ids]
        return ''.join(tokens).replace('</w>', ' ').strip()

# --- Компоненты Трансформера на PyTorch ---

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model; self.num_heads = num_heads; self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model); self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model); self.W_o = nn.Linear(d_model, d_model)
    def forward(self, q, k, v, mask=None):
        q, k, v = self.W_q(q), self.W_k(k), self.W_v(v)
        q = q.view(q.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(k.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(v.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(context.shape[0], -1, self.d_model)
        return self.W_o(context)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        pos = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionWiseFeedForward(d_model, d_ff)
    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model); self.norm3 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionWiseFeedForward(d_model, d_ff)
    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask)
        x = x + self.cross_attn(self.norm2(x), enc_output, enc_output, src_mask)
        x = x + self.ff(self.norm3(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
    def make_src_mask(self, src):
        return (src != self.pad_id).unsqueeze(1).unsqueeze(2)
    def make_tgt_mask(self, tgt):
        pad_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.shape[1]
        seq_mask = torch.tril(torch.ones((seq_len, seq_len))).bool()
        return pad_mask & seq_mask
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src = self.pos_encoder(self.embedding(src))
        tgt = self.pos_encoder(self.embedding(tgt))
        for layer in self.encoder_layers: src = layer(src, src_mask)
        for layer in self.decoder_layers: tgt = layer(tgt, src, src_mask, tgt_mask)
        return self.fc_out(tgt)

# --- Подготовка данных (используем наш BPE токенизатор) ---
def augment_data(conversations):
    # ... (код аугментации без изменений) ...
    augmented = []
    for q, a in conversations:
        augmented.append((q, a))
        words = q.split()
        if len(words) > 1:
            for i in range(len(words)):
                new_q = " ".join(words[:i] + words[i+1:])
                if new_q: augmented.append((new_q, a))
    print(f"Аугментация завершена. Исходных примеров: {len(conversations)}, стало: {len(augmented)}")
    return augmented

conversations = [
    ("привет", "здравствуй"), ("добрый день", "и вам добрый"), ("здравствуй", "и тебе привет"),
    ("пока", "до скорой встречи"), ("до свидания", "всего хорошего"), ("увидимся", "еще обязательно увидимся"),
    ("кто ты", "я нейросеть текстовая модель"), ("как тебя зовут", "у меня нет имени я просто программа"),
    ("что ты умеешь делать", "я могу отвечать на простые вопросы и фразы"),
    ("для чего ты нужен", "чтобы общаться с тобой и помогать"),
    ("как дела", "все отлично спасибо что спросил"), ("большое спасибо", "не за что"),
    ("благодарю", "всегда пожалуйста")
]
augmented_conversations = augment_data(conversations)
corpus = [q for q, a in augmented_conversations] + [a for q, a in augmented_conversations]
tokenizer = BPETokenizer(vocab_size=70); tokenizer.train(corpus)
vocab_size = len(tokenizer.vocab)
PAD_ID = tokenizer.token_to_id["<pad>"]; SOS_ID = tokenizer.token_to_id["<sos>"]; EOS_ID = tokenizer.token_to_id["<eos>"]
max_seq_length = 20
def pad_sequence(tokens, max_len, pad_id): return (tokens + [pad_id] * max_len)[:max_len]
src_data, tgt_data, y_labels = [], [], []
for q, a in augmented_conversations:
    src_tokens = tokenizer.encode(q); tgt_tokens = tokenizer.encode(a)
    src_data.append(pad_sequence(src_tokens + [EOS_ID], max_seq_length, PAD_ID))
    tgt_data.append(pad_sequence([SOS_ID] + tgt_tokens, max_seq_length, PAD_ID))
    y_labels.append(pad_sequence(tgt_tokens + [EOS_ID], max_seq_length, PAD_ID))

# Конвертируем данные в Тензоры PyTorch
src_data = torch.LongTensor(src_data)
tgt_data = torch.LongTensor(tgt_data)
y_labels = torch.LongTensor(y_labels)

# --- Обучение на PyTorch ---
d_model=128; num_heads=4; num_layers=3; d_ff=512; learning_rate=0.0001; epochs=500
model = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff, PAD_ID)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID) # Игнорируем паддинг при подсчете потерь
print("Начало обучения на PyTorch...");

model.train() # Переводим модель в режим обучения
for epoch in range(epochs):
    optimizer.zero_grad() # Очищаем градиенты
    output = model(src_data, tgt_data)
    # PyTorch CrossEntropyLoss ожидает (N, C, ...) и (N, ...)
    loss = criterion(output.view(-1, vocab_size), y_labels.view(-1))
    loss.backward() # МАГИЯ: Автоматический backpropagation
    optimizer.step() # Обновляем ВСЕ веса с помощью Adam
    if (epoch + 1) % 50 == 0:
        print(f"Эпоха {epoch+1}/{epochs}, Потери: {loss.item():.4f}")

print("Обучение завершено.");

# --- Финальная функция для общения ---
def chat(user_input, temperature=1.0, top_p=0.9):
    model.eval() # Переводим модель в режим инференса
    src_tokens = tokenizer.encode(user_input)
    src_tensor = torch.LongTensor([pad_sequence(src_tokens + [EOS_ID], max_seq_length, PAD_ID)])
    output_ids = [SOS_ID]

    for _ in range(max_seq_length - 1):
        with torch.no_grad(): # Отключаем вычисление градиентов для скорости
            tgt_tensor = torch.LongTensor([pad_sequence(output_ids, max_seq_length, PAD_ID)])
            output = model(src_tensor, tgt_tensor)
        
        last_logits = output[0, len(output_ids) - 1, :]
        last_logits /= temperature
        probs = torch.softmax(last_logits, dim=-1)
        
        # Top-p семплирование
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0
        
        if torch.sum(probs) > 0:
            probs /= torch.sum(probs)
            next_word_id = torch.multinomial(probs, num_samples=1).item()
        else:
            next_word_id = torch.argmax(last_logits).item()
            
        if next_word_id == EOS_ID: break
        output_ids.append(next_word_id)
        
    raw_response = tokenizer.decode(output_ids)
    return raw_response.replace("<sos>", "").strip()

# --- Интерактивный чат ---
print("\nМодель обучена на PyTorch. Спросите у нее 'кто ты' или 'что ты умеешь'.")
while True:
    user_message = input("Вы: ")
    if user_message.lower() == 'выход': break
    response = chat(user_message)
    print(f"Бот: {response}")
