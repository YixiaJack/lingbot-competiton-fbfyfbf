"""Train a small CNN: wrist-camera crop (96x128) -> P(this gripper is holding a tool handle).
usage: train.py OUT_DIR [epochs=10]
train = official task-3 demos (all 216) + rollouts 105a2d09 episodes < 172; test = rollouts episodes >= 172 (held out)."""
import sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

G = './graspnet'
OUT = sys.argv[1]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
torch.manual_seed(0); np.random.seed(0)
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'

Xd, Md = np.load(f'{G}/demo_X.npy'), np.load(f'{G}/demo_meta.npy')
Xr, Mr = np.load(f'{G}/roll_X.npy'), np.load(f'{G}/roll_meta.npy')
tr_r = Mr[:, 2] < 172
Xtr = np.concatenate([Xd, Xr[tr_r]]); ytr = np.concatenate([Md[:, 0], Mr[tr_r, 0]]).astype(np.float32)
Xte, yte, Mte = Xr[~tr_r], Mr[~tr_r, 0].astype(np.float32), Mr[~tr_r]
print(f'train {len(Xtr)} (hold {int(ytr.sum())}) | test {len(Xte)} (hold {int(yte.sum())}) | device {dev}', flush=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [3, 24, 48, 64, 96, 128]
        self.convs = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1, bias=False) for i in range(5)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(c) for c in ch[1:]])
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(ch[-1], 1)

    def forward(self, x):                      # x: (B, 3, 96, 128) float in [0, 1]
        x = (x - 0.45) / 0.25
        for c, b in zip(self.convs, self.bns):
            x = F.relu(b(c(x)))
        return self.fc(self.drop(x.mean((2, 3)))).squeeze(1)


def to_t(X):
    return torch.from_numpy(X).permute(0, 3, 1, 2).float().div(255)


def augment(x):
    B = x.shape[0]
    x = x * (0.75 + 0.5 * torch.rand(B, 1, 1, 1, device=x.device)) + 0.1 * (torch.rand(B, 1, 1, 1, device=x.device) - 0.5)
    x = x * (0.9 + 0.2 * torch.rand(B, 3, 1, 1, device=x.device))                       # colour cast
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    pad = F.pad(x, (8, 8, 8, 8), mode='replicate')
    dx, dy = np.random.randint(0, 17), np.random.randint(0, 17)
    return pad[:, :, dy:dy + 96, dx:dx + 128].clamp(0, 1)


net = Net().to(dev)
print('params', sum(p.numel() for p in net.parameters()), flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
steps = EPOCHS * (len(Xtr) // 256)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)


def evaluate(X, y, bs=1024):
    net.eval(); ps = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            ps.append(torch.sigmoid(net(to_t(X[i:i + bs]).to(dev))).cpu().numpy())
    net.train()
    p = np.concatenate(ps)
    pred = p >= 0.5
    acc = (pred == (y == 1)).mean()
    tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum(); tn = (~pred & (y == 0)).sum()
    return p, acc, (int(tp), int(fp), int(fn), int(tn))


for ep in range(EPOCHS):
    t0 = time.time(); perm = np.random.permutation(len(Xtr)); tot = 0
    for i in range(0, len(perm) - 255, 256):
        b = perm[i:i + 256]
        x = augment(to_t(Xtr[b]).to(dev)); y = torch.from_numpy(ytr[b]).to(dev)
        loss = F.binary_cross_entropy_with_logits(net(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
    _, acc, cm = evaluate(Xte, yte)
    print(f'epoch {ep + 1}: train loss {tot / len(perm):.4f} | held-out rollouts acc {acc:.4f} (tp fp fn tn {cm}) | {time.time() - t0:.0f} s', flush=True)

net = net.cpu().eval()
torch.save(net.state_dict(), f'{OUT}/graspnet.pt')
ex = torch.jit.trace(net, torch.zeros(1, 3, 96, 128))
ex.save(f'{OUT}/graspnet.torchscript')
net = net.to(dev)
p, acc, cm = evaluate(Xte, yte)
np.save(f'{OUT}/test_pred.npy', np.c_[Mte, p])
print('final held-out acc', round(float(acc), 4), cm)
