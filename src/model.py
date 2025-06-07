import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

# Global device declaration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

    def forward(self, x, t):
        t = t.unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(*t.shape[:2], self.d_model, device=device)
        pe[..., 0::2] = torch.sin(t * div_term)
        pe[..., 1::2] = torch.cos(t * div_term)
        x = x + pe[:x.size(0)]
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(config.emb_dim, config.dropout, config.seq_len)
        encoder_layers = nn.TransformerEncoderLayer(config.emb_dim, config.num_head, config.hid_dim, config.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, config.nlayers)
        self.seq_len = config.seq_len
        self.ninp = config.emb_dim
        self.encoder = nn.Linear(3, config.emb_dim, bias=False)
        self.decoder = nn.Linear(config.emb_dim, config.z_dim * 2)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def encode(self, x, x_mask=None):
        x = x.transpose(1, 0).to(device)
        if x_mask is None:
            x_mask = subsequent_mask(len(x))
        t = torch.cumsum(x[..., -1], 0)
        x = self.encoder(x) * math.sqrt(self.ninp)
        x = self.pos_encoder(x, t)
        output = self.transformer_encoder(x, x_mask)
        output = self.decoder(output)
        output = output[-1]
        m, v_ = torch.chunk(output, 2, dim=-1)
        v = F.softplus(v_) + 1e-5
        return m, v

class Decoder(nn.Module):
    def __init__(self, config, out_dim, softplus=False):
        super().__init__()
        self.softplus = softplus
        layers = [nn.Linear(config.z_dim, config.hid_dim), nn.ELU()]
        for _ in range(config.decoder_n_layer - 1):
            layers.extend([nn.Linear(config.hid_dim, config.hid_dim), nn.ELU()])
        layers.append(nn.Linear(config.hid_dim, out_dim))
        self.net = nn.Sequential(*layers).to(device)

    def decode(self, z):
        z = z.to(device)
        out = self.net(z)
        if self.softplus:
            out = F.softplus(out) + 1e-5
        return out

def kl_normal(qm, qv, pm, pv):
    return 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1).sum(-1)

def sample_gaussian(m, v):
    return torch.randn_like(m, device=device) * torch.sqrt(v) + m

def ll_no_events(w_i, b_i, tn_ti, t_ti):
    return torch.sum(w_i / b_i * (torch.exp(-b_i * t_ti) - torch.exp(-b_i * tn_ti)), -1)

def log_ft(t_ti, tn_ti, w_i, b_i):
    return ll_no_events(w_i, b_i, tn_ti, t_ti) + torch.log(t_intensity(w_i, b_i, t_ti))

def t_intensity(w_i, b_i, t_ti):
    return torch.sum(w_i * torch.exp(-b_i * t_ti), -1)

def s_intensity(w_i, b_i, t_ti, s_diff, inv_var):
    v_i = w_i * torch.exp(-b_i * t_ti)
    v_i = v_i / torch.sum(v_i, -1, keepdim=True)
    g2 = torch.sum(s_diff * inv_var * s_diff, -1)
    g2 = torch.sqrt(torch.prod(inv_var, -1)) * torch.exp(-0.5 * g2) / (2 * np.pi)
    return torch.sum(g2 * v_i, -1)

def intensity(w_i, b_i, t_ti, s_diff, inv_var):
    return t_intensity(w_i, b_i, t_ti) * s_intensity(w_i, b_i, t_ti, s_diff, inv_var)

class DeepSTPP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb_dim = config.emb_dim
        self.hid_dim = config.hid_dim
        self.enc = Encoder(config).to(device)
        output_dim = config.seq_len + config.num_points
        self.w_dec = Decoder(config, output_dim, softplus=True)
        self.b_dec = Decoder(config, output_dim)
        self.s_dec = Decoder(config, output_dim * 2, softplus=True)
        self.z_prior_m = nn.Parameter(torch.zeros(1, device=device), requires_grad=False)
        self.z_prior_v = nn.Parameter(torch.ones(1, device=device), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)
        self.num_points = config.num_points
        self.background = nn.Parameter(torch.rand((self.num_points, 2), device=device), requires_grad=True)
        self.optimizer = self.set_optimizer(config.opt, config.lr, config.momentum)
        self.to(device)

    def set_optimizer(self, opt, lr, momentum):
        return torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum) if opt == 'SGD' else torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, st_x):
        if self.config.sample:
            qm, qv = self.enc.encode(st_x)
            z = sample_gaussian(qm, qv)
        else:
            qm, qv = None, None
            z, _ = self.enc.encode(st_x)
        w_i = self.w_dec.decode(z)
        raw_b = self.b_dec.decode(z)
        if self.config.constrain_b == 'tanh':
            b_i = torch.tanh(raw_b) * self.config.b_max
        elif self.config.constrain_b == 'sigmoid':
            b_i = torch.sigmoid(raw_b) * self.config.b_max
        elif self.config.constrain_b == 'neg-sigmoid':
            b_i = -torch.sigmoid(raw_b) * self.config.b_max
        elif self.config.constrain_b == 'softplus':
            b_i = F.softplus(raw_b)
        elif self.config.constrain_b == 'clamp':
            b_i = torch.clamp(raw_b, -self.config.b_max, self.config.b_max)
        else:
            b_i = raw_b
        s_i = self.s_dec.decode(z) + self.config.s_min
        s_x, s_y = torch.chunk(s_i, 2, dim=-1)
        inv_var = torch.stack((1 / s_x, 1 / s_y), -1)
        return [qm, qv], w_i, b_i, inv_var

    def loss(self, st_x, st_y):
        batch = st_x.shape[0]
        background = self.background.unsqueeze(0).repeat(batch, 1, 1)
        s_diff = st_y[..., :2] - torch.cat((st_x[..., :2], background), 1)
        t_cum = torch.cumsum(st_x[..., 2], -1)
        tn_ti = t_cum[..., -1:] - t_cum
        tn_ti = torch.cat((tn_ti, torch.zeros(batch, self.num_points, device=device)), -1)
        t_ti = tn_ti + st_y[..., 2]
        [qm, qv], w_i, b_i, inv_var = self(st_x)
        sll = torch.log(s_intensity(w_i, b_i, t_ti, s_diff, inv_var))
        tll = log_ft(t_ti, tn_ti, w_i, b_i)
        if self.config.sample:
            kl = kl_normal(qm, qv, *self.z_prior).mean()
            nelbo = kl - self.config.beta * (sll.mean() + tll.mean())
        else:
            nelbo = - (sll.mean() + tll.mean())
        return nelbo, sll, tll

def calc_lamb(model, test_loader, config, scales=np.ones(3), biases=np.zeros(3),
              t_nstep=201, x_nstep=101, y_nstep=101, total_time=None, round_time=True,
              xmax=None, xmin=None, ymax=None, ymin=None):

    st_xs, st_ys, st_x_cums, st_y_cums = [], [], [], []

    for data in test_loader:
        st_x, st_y, st_x_cum, st_y_cum, (idx, _) = data
        mask = idx == 0
        st_xs.append(st_x[mask])
        st_ys.append(st_y[mask])
        st_x_cums.append(st_x_cum[mask])
        st_y_cums.append(st_y_cum[mask])
        if not torch.any(mask):
            break

    st_x = torch.cat(st_xs, 0).to(device)
    st_y = torch.cat(st_ys, 0).to(device)
    st_x_cum = torch.cat(st_x_cums, 0).to(device)
    st_y_cum = torch.cat(st_y_cums, 0).to(device)

    if total_time is None:
        total_time = st_y_cum[-1, -1, -1].item()

    print(f'Intensity time range : {total_time}')
    lambs = []

    # Spatial grid setup
    if xmax is None:
        xmax, xmin, ymax, ymin = 1.0, 0.0, 1.0, 0.0
    else:
        xmax = (xmax - biases[0]) / scales[0]
        xmin = (xmin - biases[0]) / scales[0]
        ymax = (ymax - biases[1]) / scales[1]
        ymin = (ymin - biases[1]) / scales[1]

    x_step = (xmax - xmin) / (x_nstep - 1)
    y_step = (ymax - ymin) / (y_nstep - 1)
    # x_range = torch.arange(xmin, xmax + 1e-10, x_step, device=device)
    # y_range = torch.arange(ymin, ymax + 1e-10, y_step, device=device)
    # s_grids = torch.stack(torch.meshgrid(x_range, y_range, indexing='ij'), dim=-1).view(-1, 2)
    x_range = torch.linspace(xmin, xmax, x_nstep, device=device)
    y_range = torch.linspace(ymin, ymax, y_nstep, device=device)
    s_grids = torch.stack(torch.meshgrid(x_range, y_range, indexing='ij'), dim=-1).view(-1, 2)

    # Temporal grid setup
    t_start = st_x_cum[0, -1, -1].item()
    t_step = (total_time - t_start) / (t_nstep - 1)
    if round_time:
        t_range = torch.arange(round(t_start)+1, round(total_time), 1.0, device=device)
    else:
        t_range = torch.arange(t_start, total_time, t_step, device=device)

    background = model.background.unsqueeze(0).detach()

    with torch.no_grad():
        _, w_i, b_i, inv_var = model(st_x)
        w_i, b_i, inv_var = w_i.detach(), b_i.detach(), inv_var.detach()

    his_st     = torch.vstack((st_x[0], st_y.squeeze())).cpu().numpy()
    his_st_cum = torch.vstack((st_x_cum[0], st_y_cum.squeeze())).cpu().numpy()

    for t in tqdm(t_range):
        i = torch.sum(st_x_cum[:, -1, -1] <= t).item() - 1
        st_x_ = st_x[i:i+1]
        w_i_ = w_i[i:i+1]
        b_i_ = b_i[i:i+1]
        inv_var_ = inv_var[i:i+1]

        t_ = (t - st_x_cum[i:i+1, -1, -1] - biases[-1]) / scales[-1]
        t_cum = torch.cumsum(st_x_[..., -1], -1)
        tn_ti = t_cum[..., -1:] - t_cum
        tn_ti = torch.cat((tn_ti, torch.zeros(1, config.num_points, device=device)), -1)
        t_ti = tn_ti + t_

        lamb_t = t_intensity(w_i_, b_i_, t_ti) / np.prod(scales)

        N = len(s_grids)
        s_x_ = torch.cat((st_x_[..., :-1], background), 1).repeat(N, 1, 1)
        s_diff = s_grids.unsqueeze(1) - s_x_
        lamb_s = s_intensity(w_i_.repeat(N, 1), b_i_.repeat(N, 1), t_ti.repeat(N, 1),
                             s_diff, inv_var_.repeat(N, 1, 1))

        lamb = (lamb_s * lamb_t).view(x_nstep, y_nstep)
        lambs.append(lamb.cpu().numpy())  # only move final output to CPU

    return (
        lambs,
        x_range.cpu().numpy() * scales[0] + biases[0],
        y_range.cpu().numpy() * scales[1] + biases[1],
        t_range.cpu().numpy(),
        his_st_cum[:, :2],
        his_st_cum[:, 2]
    )
