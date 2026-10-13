import numpy as np
import sys
import os

blok = 40
neur_ukryt = 8
il_epoch = 10
uczenie = 0.1

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from szyfry import szyfСezar

def prepare_dataset(file_path, seq_length=blok, shift=3):
    with open(file_path, "r") as f:
        text = f.read()
    text = " ".join(text.split())
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    inputs, targets = [], []
    for i in range(0, len(text) - seq_length, seq_length):
        original_chunk = text[i:i + seq_length]
        cipher_chunk = szyfСezar(original_chunk, shift)
        inputs.append([char_to_idx[ch] for ch in cipher_chunk])
        targets.append([char_to_idx[ch] for ch in original_chunk])
        
    return inputs, targets, vocab_size, char_to_idx, idx_to_char

class CaesarRNN:
    def __init__(self, vocab_size, hidden_size=neur_ukryt):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, vocab_size) * uczenie
        self.W_hh = np.random.randn(hidden_size, hidden_size) * uczenie
        self.W_hy = np.random.randn(vocab_size, hidden_size) * uczenie
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((vocab_size, 1))
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0, keepdims=True)
    def forward_backward(self, inputs, targets, learning_rate=uczenie):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.zeros((self.hidden_size, 1))
        loss = 0
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.W_xh, xs[t]) + np.dot(self.W_hh, hs[t-1]) + self.b_h)
            ys[t] = np.dot(self.W_hy, hs[t]) + self.b_y
            ps[t] = self.softmax(ys[t])
            loss += -np.log(ps[t][targets[t], 0] + 1e-8)
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        dh_next = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dW_hy += np.dot(dy, hs[t].T)
            db_y += dy
            dh = np.dot(self.W_hy.T, dy) + dh_next
            dh_raw = (1 - hs[t] ** 2) * dh
            db_h += dh_raw
            dW_xh += np.dot(dh_raw, xs[t].T)
            dW_hh += np.dot(dh_raw, hs[t-1].T)
            dh_next = np.dot(self.W_hh.T, dh_raw)
        for param in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(param, -5, 5, out=param)
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y
        return loss
    def predict(self, inputs, idx_to_char):
        h_prev = np.zeros((self.hidden_size, 1))
        result = []
        for t in range(len(inputs)):
            x = np.zeros((self.vocab_size, 1))
            x[inputs[t]] = 1
            h = np.tanh(np.dot(self.W_xh, x) + np.dot(self.W_hh, h_prev) + self.b_h)
            y = np.dot(self.W_hy, h) + self.b_y
            e_x = np.exp(y - np.max(y))
            p = e_x / e_x.sum(axis=0, keepdims=True)
            pred_idx = np.argmax(p)
            result.append(idx_to_char[pred_idx])
            h_prev = h
        return "".join(result)

if __name__ == "__main__":
    file_path = "../../book_cleaned.txt"  
    if os.path.exists(file_path):
        inputs, targets, vocab_size, c2i, i2c = prepare_dataset(file_path, seq_length=blok, shift=3)
        print(f"Bloków po {blok} symbolów: {len(inputs)}")
        print(f"Neuronów: {neur_ukryt}")
        rnn = CaesarRNN(vocab_size=vocab_size, hidden_size=neur_ukryt)
        sample_input = inputs[0]
        sample_target_str = "".join([i2c[idx] for idx in targets[0]])
        sample_input_str = "".join([i2c[idx] for idx in sample_input])

        print(f"\n[Początek uczenia]")
        print(f"Tekst szyfrowany: {sample_input_str}")
        print(f"Original:         {sample_target_str}")
        print(f"Odpowiedź modelu: {rnn.predict(sample_input, i2c)}")
        for epoch in range(0, il_epoch):
            total_loss = 0
            num_batches = min(100, len(inputs))
            for i in range(num_batches):
                loss = rnn.forward_backward(inputs[i], targets[i], learning_rate=0.03)
                total_loss += loss
            avg_loss = total_loss / num_batches
            print(f"Epoka: {epoch+1} | Średnii bląd: {avg_loss:.4f} | Odpowiedź modelu: {rnn.predict(sample_input, i2c)}")
        print(f"\n[Koniec uczenia]")
        print(f"Tekst szyfrowany: {sample_input_str}")
        print(f"Original:         {sample_target_str}")
        print(f"Odpowiedź modelu: {rnn.predict(sample_input, i2c)}")
    else:
        print("Brak pliku!")