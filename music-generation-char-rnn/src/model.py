"""Model builders (PRD 6.1-6.2). All share Emb(256)->head; sizes from config."""
import torch
import torch.nn as nn


class CharRNN(nn.Module):  # main many-to-one: Emb256 + 3xLSTM256 + Drop + Dense
    def __init__(self, vocab, emb=256, units=(256, 256, 256), dropout=0.3,
                 batchnorm=False):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.l1 = nn.LSTM(emb, units[0], batch_first=True)
        self.l2 = nn.LSTM(units[0], units[1], batch_first=True)
        self.l3 = nn.LSTM(units[1], units[2], batch_first=True)
        # ponytail: one BN slot only - the E4 variant; main path stays clean
        self.bn = nn.LayerNorm(units[2]) if batchnorm else None
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(units[2], vocab)

    def forward(self, x, h=None, return_state=False):
        e = self.emb(x)
        hs, o = [], e
        for i, l in enumerate((self.l1, self.l2, self.l3)):
            o, hn = l(o, None if h is None else h[i])
            hs.append(hn)
            # ponytail: detach happens in the training loop, not here
        o = o[:, -1, :]
        if self.bn is not None:
            o = self.bn(o)
        out = self.fc(self.drop(o))
        return (out, hs) if return_state else out


class ManyToManyRNN(nn.Module):  # variant B: prediction at every step
    def __init__(self, vocab, emb=256, units=(256, 256, 256), dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.l1 = nn.LSTM(emb, units[0], batch_first=True)
        self.l2 = nn.LSTM(units[0], units[1], batch_first=True)
        self.l3 = nn.LSTM(units[1], units[2], batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(units[2], vocab)

    def forward(self, x, h=None):
        e = self.emb(x)
        o, _ = self.l1(e)
        o, _ = self.l2(o)
        o, _ = self.l3(o)
        return self.fc(self.drop(o))  # (B, T, V)


class StatefulRNN(CharRNN):  # variant C: same arch; statefulness in train loop
    pass


class CharCNN(nn.Module):  # Phase 6: Emb + Conv1D k5/k3 ReLU + pool + Dense
    def __init__(self, vocab, emb=256, filters=(256, 256), kernels=(5, 3),
                 dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.c1 = nn.Conv1d(emb, filters[0], kernels[0], padding="same")
        self.c2 = nn.Conv1d(filters[0], filters[1], kernels[1], padding="same")
        self.relu = nn.ReLU()  # Unit II ReLU evidence lives here (E5/E10)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(filters[1], 256)
        self.fc2 = nn.Linear(256, vocab)

    def forward(self, x, h=None):
        e = self.emb(x).transpose(1, 2)
        o = self.relu(self.c1(e))
        o = self.relu(self.c2(o))
        o = o.max(dim=2).values  # global max pool over time
        return self.fc2(self.drop(self.relu(self.fc1(o))))


class SmallRNN(nn.Module):  # Phase 3 baseline (c): vanilla Elman RNN, small
    def __init__(self, vocab, emb=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.rnn = nn.RNN(emb, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, x, h=None):
        _, h = self.rnn(self.emb(x))
        return self.fc(h[-1])


def build(name, vocab, **kw):
    if name == "small_rnn":  # baseline: fixed tiny shape, ignores main-model kw
        return SmallRNN(vocab, emb=kw.get("emb", 64),
                        hidden=kw.get("hidden", 128))
    if name in ("many2many", "cnn"):  # no BN slot on these variants (E4 is main-only)
        kw = {k: v for k, v in kw.items() if k != "batchnorm"}
    return {"main": CharRNN, "many2many": ManyToManyRNN,
            "stateful": StatefulRNN, "cnn": CharCNN}[name](
            vocab, **{k: v for k, v in kw.items()
                      if k in ("emb", "units", "dropout", "batchnorm",
                               "filters", "kernels")})


def count_params(m):
    return sum(p.numel() for p in m.parameters())
