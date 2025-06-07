import math
import numpy as np
from mamba_ssm import Mamba
import torch
from torch import nn
from torch.nn import functional as F

from tqdm.auto import tqdm

"""
Return a square attention mask to only allow self-attention layers to attend the earlier positions
"""
def subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))



"""
Injects some information about the relative or absolute position of the tokens in the sequence
ref: https://github.com/harvardnlp/annotated-transformer/
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

    def forward(self, x, t):
        pe = torch.zeros(self.max_len, self.d_model, device=x.device)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * (-math.log(10000.0) / self.d_model))

        t = t.unsqueeze(-1)
        pe = torch.zeros(*t.shape[:2], self.d_model, device=x.device)
        pe[..., 0::2] = torch.sin(t * div_term)
        pe[..., 1::2] = torch.cos(t * div_term)

        x = x + pe[:x.size(0)]
        return self.dropout(x)


class Time2Vec(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.linear = nn.Linear(in_features, 1)
        self.freqs = nn.Linear(in_features, out_features - 1)

    def forward(self, t):
        # t: [batch, seq_len, 1] assumed
        v_linear = self.linear(t)
        v_periodic = torch.sin(self.freqs(t))
        return torch.cat([v_linear, v_periodic], dim=-1)
    
"""
Encode time/space record to variational posterior for location latent
"""

class Encoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.emb_dim = config.emb_dim

        self.time_emb = Time2Vec(1, self.emb_dim)

        # s (2D), delta_t (1D), t_emb (emb_dim)
        self.input_proj = nn.Linear(2 + 1 + self.emb_dim, self.emb_dim)

        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config.emb_dim),
                Mamba(
                    d_model=config.emb_dim,
                    d_state=config.mamba_d_state,
                    expand=config.mamba_expand,
                    bias=config.mamba_bias
                )
            )
            for _ in range(config.nlayers)
        ])

        # self.encoder = nn.Sequential(
        #     *[Mamba(
        #         d_model=config.emb_dim,
        #         d_state=config.mamba_d_state,
        #         expand=config.mamba_expand,
        #         bias=config.mamba_bias
        #     ) for _ in range(config.n_layers)]
        # )

        
        self.decoder = nn.Linear(config.emb_dim, config.z_dim * 2)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward_encoder(self, x):
        for layer in self.encoder_layers:
            x = x + layer(x)  # Residual connection
        return x

    def encode(self, x, x_mask=None):
        s = x[..., :2]          # [batch, seq_len, 2]
        t = x[..., 2:]          # [batch, seq_len, 1]

        # Δt feature
        delta_t = t - torch.cat([torch.zeros_like(t[:, :1]), t[:, :-1]], dim=1)  # [B, T, 1]

        # Time2Vec embedding
        t_encoded = self.time_emb(t)  # [B, T, emb_dim]

        # Concatenate input features
        input = torch.cat([s, delta_t, t_encoded], dim=-1)  # [B, T, 2+1+emb_dim]
        input = self.input_proj(input)  # [B, T, emb_dim]

        # Pass through Mamba blocks
        output = self.forward_encoder(input)

        last_output = output[:, -1, :]
        stats = self.decoder(last_output)
        m, v_ = torch.chunk(stats, 2, dim=-1)
        v = F.softplus(v_) + 1e-5
        return m, v


"""
Decode latent variable to spatiotemporal kernel coefficients
"""
class Decoder(nn.Module):
    def __init__(self, config, out_dim, softplus=False):
        super().__init__()
        self.z_dim = config.z_dim
        self.softplus = softplus
        self.net = nn.Sequential(
            nn.Linear(config.z_dim, config.hid_dim),
            nn.GELU(),
            nn.Linear(config.hid_dim, config.hid_dim),
            nn.GELU(),
            nn.Linear(config.hid_dim, config.hid_dim),
            nn.GELU(),
            nn.Linear(config.hid_dim, out_dim)
        )


    def decode(self, z):
        output = self.net(z)
        if self.softplus:
            output = F.softplus(output) + 1e-5
        return output

    
def kl_normal(qm, qv, pm, pv):
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)  # Summing over the correct dimension
    return kl



def sample_gaussian(m, v):
    z = torch.randn_like(m, device=m.device)
    return z * torch.sqrt(v) + m




"""
Log likelihood of no events happening from t_n to t
- ∫_{t_n}^t λ(t') dt' 

tn_ti: [batch, seq_len]
t_ti: [batch, seq_len]
w_i: [batch, seq_len]
b_i: [batch, seq_len]

return: scalar
"""
def ll_no_events(w_i, b_i, tn_ti, t_ti):
    return torch.sum(w_i / b_i * (torch.exp(-b_i * t_ti) - torch.exp(-b_i * tn_ti)), -1)


# def log_ft(t_ti, tn_ti, w_i, b_i):
#     return ll_no_events(w_i, b_i, tn_ti, t_ti) + torch.log(t_intensity(w_i, b_i, t_ti))
def log_ft(t_ti, tn_ti, w_i, b_i):
    ll = ll_no_events(w_i, b_i, tn_ti, t_ti)
    return ll + t_intensity(w_i, b_i, t_ti)


"""
Compute spatial/temporal/spatiotemporal intensities

tn_ti: [batch, seq_len]
s_diff: [batch, seq_len, 2]
inv_var = [batch, seq_len, 2]
w_i: [batch, seq_len]
b_i: [batch, seq_len]

return: λ(t) [batch]
return: f(s|t) [batch] 
return: λ(s,t) [batch]
"""
def safe_exp(x, max_val=50):
    return torch.exp(torch.clamp(x, max=-max_val, min=max_val))

def t_intensity(w_i, b_i, t_ti):
    # exponent = -b_i * t_ti
    # exponent = -b_i * t_ti
    # v_i = w_i * safe_exp(exponent)  
    # return torch.sum(v_i, -1)

    log_v = torch.log(w_i + 1e-8) - b_i * t_ti
    return torch.logsumexp(log_v, dim=-1)


def s_intensity(w_i, b_i, t_ti, s_diff, inv_var):
    v_i = w_i * torch.exp(-b_i * t_ti)
    v_i = v_i / torch.sum(v_i, -1).unsqueeze(-1)  # normalize
    g2 = torch.sum(s_diff * inv_var * s_diff, -1)
    
    # Clamp the exponent to avoid extreme values
    g2 = torch.clamp(g2, min=-20, max=20)  # Adjust as needed
    
    # Safe exponential
    coef = torch.sqrt(torch.prod(inv_var, -1))
    ker = coef * torch.exp(-0.5 * g2) / (2 * math.pi)
    # print("Spatial kernel (ker) values:", ker.min(), ker.max())
    f_s_cond_t = torch.sum(ker * v_i, -1)
    return f_s_cond_t



def intensity(w_i, b_i, t_ti, s_diff, inv_var):
    return t_intensity(w_i, b_i, t_ti) * s_intensity(w_i, b_i, t_ti, s_diff, inv_var)


"""
STPP model with VAE: directly modeling λ(s,t)
"""
class DeepSTPP(nn.Module):
    def __init__(self, config, device):
        super(DeepSTPP, self).__init__()
        self.config = config
        self.emb_dim = config.emb_dim
        self.hid_dim = config.hid_dim
        self.device = device
        self.current_epoch = 0 
        
        # VAE for predicting spatial intensity
        self.enc = Encoder(config, device)
        
        output_dim = config.seq_len + config.num_points
        self.w_dec = Decoder(config, output_dim, softplus=True)
        self.b_dec = Decoder(config, output_dim)
        self.s_dec = Decoder(config, output_dim * 2, softplus=True)
        
        # Set prior as fixed parameter attached to Module
        self.z_prior_m = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)
        
        # Background 
        self.num_points = config.num_points
        self.background = nn.Parameter(torch.rand((self.num_points, 2)), requires_grad=True)
        
        self.optimizer = self.set_optimizer(config.opt, config.lr, config.momentum)
        self.to(device)


    """
    st_x: [batch, seq_len, 3] (lat, lon, time)
    st_y: [batch, 1, 3]
    """
    def loss(self, st_x, st_y):
        batch = st_x.shape[0]
        background = self.background.unsqueeze(0).repeat(batch, 1, 1)
        
        s_diff = st_y[..., :2] - torch.cat((st_x[..., :2], background), 1) # s - s_i
        t_cum = torch.cumsum(st_x[..., 2], -1)
        
        tn_ti = t_cum[..., -1:] - t_cum # t_n - t_i
        # tn_ti = torch.cat((tn_ti, torch.zeros(batch, self.num_poinWts).to(self.device)), -1)
        tn_ti = torch.cat((tn_ti, torch.zeros(batch, self.num_points, device=st_x.device)), -1)
        t_ti  = tn_ti + st_y[..., 2] # t - t_i

        [qm, qv], w_i, b_i, inv_var = self(st_x)
            
        # Calculate likelihood
        sll = torch.log(s_intensity(w_i, b_i, t_ti, s_diff, inv_var))
        tll = log_ft(t_ti, tn_ti, w_i, b_i)
        
        # KL Divergence
        if self.config.sample:
            kl = kl_normal(qm, qv, *self.z_prior).mean()
            kl_weight = min(1.0, self.current_epoch / self.config.kl_warmup_epochs)
            nelbo = kl * kl_weight - self.config.beta * (sll.mean() + tll.mean())
        else:
            nelbo = - (sll.mean() + tll.mean())


        return nelbo, sll, tll
   
    
    def forward(self, st_x):        
        # Encode history locations and times
        if self.config.sample:
            qm, qv = self.enc.encode(st_x) # Variational posterior
            # Monte Carlo
            z = sample_gaussian(qm, qv)
        else:
            qm, qv = None, None
            z, _ = self.enc.encode(st_x)
        
        w_i = self.w_dec.decode(z)
        b_i = torch.nn.functional.softplus(self.b_dec.decode(z)) + 1e-3

                    
        s_i = self.s_dec.decode(z) + self.config.s_min
        
        s_x, s_y = torch.split(s_i, s_i.size(-1) // 2, dim=-1)
        inv_var = torch.stack((1 / s_x, 1 / s_y), -1)

        return [qm, qv], w_i, b_i, inv_var
  
    
    def set_optimizer(self, opt, lr, momentum):
        if opt == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        elif opt == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=lr)


"""
Calculate the uniformly samplded spatiotemporal intensity with a given
number of spatiotemporal steps  
"""
def calc_lamb(model, test_loader, config, device, scales=np.ones(3), biases=np.zeros(3),
              t_nstep=201, x_nstep=101, y_nstep=101, total_time=None, round_time=True,
              xmax=None, xmin=None, ymax=None, ymin=None):
    
    # Aggregate data
    st_xs = []
    st_ys = []
    st_x_cums = []
    st_y_cums = []
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

    if xmax is None:
        xmax, xmin = 1.0, 0.0
        ymax, ymin = 1.0, 0.0
    else:
        xmax = (xmax - biases[0]) / scales[0]
        xmin = (xmin - biases[0]) / scales[0]
        ymax = (ymax - biases[1]) / scales[1]
        ymin = (ymin - biases[1]) / scales[1]

    x_step = (xmax - xmin) / (x_nstep - 1)
    y_step = (ymax - ymin) / (y_nstep - 1)

    x_range = torch.arange(xmin, xmax + 1e-10, x_step, device=device)
    y_range = torch.arange(ymin, ymax + 1e-10, y_step, device=device)
    s_grids = torch.stack(torch.meshgrid(x_range, y_range, indexing='ij'), dim=-1).view(-1, 2)

    t_start = st_x_cum[0, -1, -1].item()
    t_step = (total_time - t_start) / (t_nstep - 1)
    if round_time:
        t_range = torch.arange(round(t_start)+1, round(total_time), 1.0, device=device)
    else:
        t_range = torch.arange(t_start, total_time, t_step, device=device)

    background = model.background.unsqueeze(0).to(device).detach()

    _, w_i, b_i, inv_var = model(st_x)

    his_st     = torch.vstack((st_x[0], st_y.squeeze())).cpu().numpy()
    his_st_cum = torch.vstack((st_x_cum[0], st_y_cum.squeeze())).cpu().numpy()

    for t in tqdm(t_range):
        i = sum(st_x_cum[:, -1, -1] <= t).item() - 1
        st_x_ = st_x[i:i+1]
        w_i_ = w_i[i:i+1]
        b_i_ = b_i[i:i+1]
        inv_var_ = inv_var[i:i+1]

        t_ = t - st_x_cum[i:i+1, -1, -1]
        t_ = (t_ - biases[-1]) / scales[-1]

        t_cum = torch.cumsum(st_x_[..., -1], -1)
        tn_ti = t_cum[..., -1:] - t_cum
        tn_ti = torch.cat((tn_ti, torch.zeros(1, config.num_points, device=device)), -1)
        t_ti = tn_ti + t_

        # LOG-STABLE t_intensity
        log_lamb_t = torch.log(w_i_) - b_i_ * t_ti
        log_lamb_t = torch.logsumexp(log_lamb_t, dim=-1)

        # s_intensity (still unsafe, compute safely now)
        N = len(s_grids)
        s_x_ = torch.cat((st_x_[..., :-1], background), 1).repeat(N, 1, 1)
        s_diff = s_grids.unsqueeze(1) - s_x_

        # Compute log of s_intensity (log kernel)
        g2 = torch.sum(s_diff * inv_var_.repeat(N, 1, 1) * s_diff, -1)
        g2 = torch.clamp(g2, min=-20, max=20)

        log_coef = 0.5 * torch.sum(torch.log(inv_var_.repeat(N, 1, 1)), dim=-1) - math.log(2 * math.pi)
        log_ker = log_coef - 0.5 * g2

        v_i = w_i_.repeat(N, 1) * torch.exp(-b_i_.repeat(N, 1) * t_ti.repeat(N, 1))
        v_i = v_i / torch.sum(v_i, -1).unsqueeze(-1)

        log_lamb_s = torch.logsumexp(log_ker + torch.log(v_i), dim=-1)

        # Combine log intensities
        log_lamb = (log_lamb_t + log_lamb_s).view(x_nstep, y_nstep)
        lambs.append(log_lamb.detach().cpu().numpy())

    return (
        lambs,
        x_range.cpu().numpy() * scales[0] + biases[0],
        y_range.cpu().numpy() * scales[1] + biases[1],
        t_range.cpu().numpy(),
        his_st_cum[:, :2],
        his_st_cum[:, 2]
    )
