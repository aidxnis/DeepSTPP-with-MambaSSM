import abc
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eps = 1e-10  # A negligible positive number

class SyntheticDataset(abc.ABC):
    def __init__(self, dist_only=False):
        self.his_s, self.his_t, self.t_start, self.t_end = None, None, None, None
        self.train, self.val, self.test = None, None, None
        self.st_min, self.st_max = None, None  # torch-based min-max scaling
        self.dist_only = dist_only

    @abc.abstractmethod
    def lamb_st(self, mu, his_s, his_t, s, t):
        pass

    @abc.abstractmethod
    def generate(self, t_start, t_end):
        pass

    @staticmethod
    def g0(s, s_mu, s_sqrt_inv_det_cov, s_inv_cov):
        return SyntheticDataset.g2(s, s_mu.view(1, 2), s_sqrt_inv_det_cov, s_inv_cov)

    @staticmethod
    def g1(t, his_t, alpha, beta):
        delta_t = t - his_t
        return alpha * torch.exp(-beta * delta_t)

    @staticmethod
    def g2(s, his_s, s_sqrt_inv_det_cov, s_inv_cov):
        delta_s = s - his_s  # [N,2]
        exponent = -0.5 * torch.sum((delta_s @ s_inv_cov) * delta_s, dim=-1)
        return (1 / (2 * np.pi)) * s_sqrt_inv_det_cov * torch.exp(exponent)

    def save(self, text_path):
        np.savetxt(text_path, np.hstack((self.his_s.cpu().numpy(), self.his_t.cpu().numpy().reshape(-1, 1))),
                   delimiter=',', fmt='%f')

    def load(self, text_path, t_start, t_end):
        self.t_start, self.t_end = t_start, t_end
        his_st = np.loadtxt(text_path, delimiter=',')
        his_s = torch.tensor(his_st[:, :2], dtype=torch.float32, device=DEVICE).clone().detach()
        his_t = torch.tensor(his_st[:, 2], dtype=torch.float32, device=DEVICE).clone().detach()

        idx = (his_t >= t_start) & (his_t < t_end)
        self.his_s = his_s[idx].clone().detach()
        self.his_t = his_t[idx].clone().detach()

        if isinstance(self, DEBMDataset):
            self.his_t[0] -= eps

    def dataset(self, lookback=10, lookahead=1, split=None):
        his_s = torch.as_tensor(self.his_s, dtype=torch.float32, device=DEVICE).clone().detach()
        his_t = torch.as_tensor(self.his_t, dtype=torch.float32, device=DEVICE).clone().detach()

        if self.dist_only:
            delta = his_s[1:] - his_s[:-1]
            dist = torch.norm(delta, dim=1)
            dist = torch.cat((torch.tensor([0.], device=DEVICE), dist)).view(-1, 1).clone().detach()
            st_data = torch.cat((dist, his_t.view(-1, 1)), dim=1)
        else:
            st_data = torch.cat((his_s, his_t.view(-1, 1)), dim=1)

        delta_t = torch.cat((torch.tensor([0.], device=DEVICE), his_t[1:] - his_t[:-1])).clone().detach()
        st_data[:, -1] = delta_t

        self.st_min = st_data.min(dim=0, keepdim=True).values
        self.st_max = st_data.max(dim=0, keepdim=True).values
        st_data = (st_data - self.st_min) / (self.st_max - self.st_min + 1e-6)

        num_features = 2 if self.dist_only else 3
        length = len(st_data) - lookback - lookahead

        st_input = torch.stack([st_data[i:i + lookback] for i in range(length)]).clone().detach()
        st_label = torch.stack([st_data[i + lookback:i + lookback + lookahead] for i in range(length)]).clone().detach()

        if split is None:
            split = [8, 1, 1]
        split = torch.tensor(split, dtype=torch.float32)
        split = split / split.sum()
        train_size = int(split[0] * length)
        test_size = int(split[2] * length)

        self.train = TensorDataset(st_input[:train_size], st_label[:train_size])
        self.val = TensorDataset(st_input[train_size:-test_size], st_label[train_size:-test_size])
        self.test = TensorDataset(st_input[-test_size:], st_label[-test_size:])

        print("Finished.")

    def get_lamb_st(self, x_num, y_num, t_num, t_start, t_end):
        mask   = (self.his_t >= t_start) & (self.his_t < t_end)
        his_s0 = self.his_s[mask]
        his_t0 = self.his_t[mask]

        # build a flat grid of N = x_num*y_num points
        x = torch.linspace(his_s0[:,0].min(), his_s0[:,0].max(), x_num, device=DEVICE)
        y = torch.linspace(his_s0[:,1].min(), his_s0[:,1].max(), y_num, device=DEVICE)
        gx, gy = torch.meshgrid(x, y, indexing='ij')
        s_grids = torch.stack([gx.flatten(), gy.flatten()], dim=1)

        t_range = torch.linspace(t_start, t_end, t_num, device=DEVICE)
        mu_vec  = torch.full((x_num*y_num,), self.mu, device=DEVICE)

        lambs = []
        for t in tqdm(t_range):
            # history up to t
            sub_mask  = his_t0 < t
            lamb_flat = self.lamb_St(mu_vec, his_s0[sub_mask], t)
            # reshape and move to CPU/NumPy
            lambs.append(lamb_flat.view(x_num, y_num).T.cpu().numpy())

        return lambs, x.cpu().numpy(), y.cpu().numpy(), t_range.cpu().numpy()


"""
Simulate Spatio-Temporal Self-Correcting Process
"""
import torch
import numpy as np
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class STSCPDataset(SyntheticDataset):
    def __init__(self, g0_cov, g2_cov, alpha, beta, mu, gamma, lamb_max=10,
                 max_history=100, x_num=51, y_num=51, dist_only=False):
        super().__init__(dist_only)
        self.g0_cov = torch.tensor(g0_cov, dtype=torch.float32)
        self.g2_cov = torch.tensor(g2_cov, dtype=torch.float32)
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.gamma = gamma
        self.lamb_max = lamb_max
        self.max_history = max_history

        self.x_num, self.y_num = x_num, y_num
        x_range = torch.linspace(0, 1, x_num)
        y_range = torch.linspace(0, 1, y_num)
        grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='ij')
        s_grids = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1)

        self.x_range = x_range.to(DEVICE)
        self.y_range = y_range.to(DEVICE)
        self.s_grids = s_grids.to(DEVICE)
        self.t_range, self.lambs = None, []

        # g0_mat
        g0_mat = self.g0(s_grids.to(DEVICE), torch.tensor([0.5, 0.5], device=DEVICE),
                         1 / torch.sqrt(torch.linalg.det(self.g0_cov.to(DEVICE))),
                         torch.linalg.inv(self.g0_cov.to(DEVICE)))
        g0_mat = g0_mat / torch.sum(g0_mat) * x_num * y_num * mu
        self.g0_mat = g0_mat  # already on DEVICE

        # g2_mats on CPU to prevent memory blowup
        num_point = x_num * y_num
        g2_mats = torch.zeros((num_point, num_point), dtype=torch.float32)  # CPU

        center = torch.tensor([0.5, 0.5])
        for i, s in enumerate(s_grids):
            g2_cov_i = self.g2_cov / (gamma * torch.sum((s - center) ** 2) + 1)
            inv_cov_i = torch.linalg.inv(g2_cov_i)
            sqrt_inv_det = 1 / torch.sqrt(torch.linalg.det(g2_cov_i))
            g2_vals = SyntheticDataset.g2(s_grids, s.unsqueeze(0), sqrt_inv_det, inv_cov_i)
            g2_vals = g2_vals / torch.sum(g2_vals) * x_num * y_num
            g2_mats[i] = g2_vals.clone().detach().cpu() 

        self.g2_mats = g2_mats.to(DEVICE)

    def stoi(self, s):
        """
        Map s ∈ [0,1]^2 to a flat index in [0, x_num*y_num).
        Works for s shaped (...,2) or (N,2).
        """
        # 1) scale to [0, x_num-1], [0, y_num-1] and round
        coords = torch.round(
            s * torch.tensor([self.x_num - 1, self.y_num - 1], device=DEVICE)
        ).long()  # now int64

        # 2) split into x_idx, y_idx
        x_idx = coords[..., 0]
        y_idx = coords[..., 1]

        # 3) compute flat index
        return x_idx + y_idx * self.y_num


    def lamb_st(self, mu, his_s, his_t,s, t):
        i = self.stoi(s)
        if his_s.nelement() == 0:
            influence = 0
        else:
            influence = torch.sum(self.g2_mats[self.stoi(his_s), i])
        val = mu[i] * torch.exp(self.g0_mat[i] * self.beta * t - self.alpha * influence)
        return torch.minimum(val, torch.tensor(self.lamb_max, device=DEVICE))

    def lamb_St(self, mu, his_s, t):
        if his_s.nelement() == 0:
            influence = 0
        else:
            influence = torch.sum(self.g2_mats[self.stoi(his_s)].reshape(-1, self.x_num * self.y_num), dim=0)
        val = mu * torch.exp(self.g0_mat * self.beta * t - self.alpha * influence)
        return torch.minimum(val, self.lamb_max * torch.ones_like(val))

    def lamb_t(self, mu, his_s, t):
        return torch.sum(self.lamb_St(mu, his_s, t)) / (self.x_num * self.y_num)

    def generate(self, t_start, t_end, t_num=None, verbose=False):
        """
        GPU‐thinned simulation of the self‑correcting process,
        mirroring your CPU version but using torch tensors.
        """
        self.t_start, self.t_end = t_start, t_end
        if t_num is None:
            t_num = self.max_history * 2

        # Initialize storage
        self.t_range = torch.tensor([t_start], device=DEVICE)
        # Start with uniform base intensity
        self.lambs   = [torch.full((self.x_num*self.y_num,), self.mu, device=DEVICE)]
        self.his_s   = torch.zeros((0, 2), device=DEVICE)
        self.his_t   = torch.tensor([], device=DEVICE)

        # Keep generating until we cover [t_start, t_end]
        while self.t_range[-1] < self.t_end:
            t0      = self.t_range[-1]
            # generate_batch yields (new_t_range, batch_lambs, new_his_s, new_his_t)
            t_rng, batch_lambs, new_s, new_t = self.generate_batch(
                t_start=t0, t_num=t_num, mu=self.lambs[-1], verbose=verbose
            )
            # append
            self.lambs   += batch_lambs
            self.t_range = torch.cat((self.t_range, t_rng.to(DEVICE)))
            self.his_s   = torch.cat((self.his_s, new_s), dim=0)
            self.his_t   = torch.cat((self.his_t, new_t), dim=0)

        # reshape each lamb vector into (x_num, y_num)
        self.lambs = [
            l.view(self.x_num, self.y_num).T.cpu().numpy()
            for l in self.lambs
        ]


    def get_lamb_st(self, x_num, y_num, t_num, t_start, t_end):
        """
        Vectorized evaluation over (x,y,t) — analogous to your CPU
        version, but using lamb_St under the hood.
        """
        # restrict history
        mask   = (self.his_t >= t_start) & (self.his_t < t_end)
        his_s0 = self.his_s[mask]
        his_t0 = self.his_t[mask]

        # build grid
        x = torch.linspace(his_s0[:,0].min(), his_s0[:,0].max(), x_num, device=DEVICE)
        y = torch.linspace(his_s0[:,1].min(), his_s0[:,1].max(), y_num, device=DEVICE)
        gx, gy   = torch.meshgrid(x, y, indexing='ij')
        s_grids  = torch.stack((gx.flatten(), gy.flatten()), dim=1)

        # time axis
        t_range = torch.linspace(t_start, t_end, t_num, device=DEVICE)
        mu_vec  = torch.full((x_num*y_num,), self.mu, device=DEVICE)

        lambs = []
        for t in tqdm(t_range, desc="Evaluating λ(s,t)"):
            sub = his_s0[his_t0 < t]
            flat = self.lamb_St(mu_vec, sub, t)
            lambs.append(flat.view(x_num, y_num).T.cpu().numpy())

        return lambs, x.cpu().numpy(), y.cpu().numpy(), t_range.cpu().numpy()
 
    
"""
Learn a spatio-temporal Hawkes process
"""
class STHPLearner:
    def __init__(self):
        self.dset = STHPDataset(
            s_mu=None,
            g0_cov=np.eye(2),
            g2_cov=np.eye(2),
            alpha=None,
            beta=None,
            mu=None
        )

    def train(self, train_s, train_t, t_start=None, t_end=None):
        self.dset.his_s = torch.as_tensor(train_s, dtype=torch.float32, device=DEVICE).clone().detach()
        self.dset.his_t = torch.as_tensor(train_t, dtype=torch.float32, device=DEVICE).clone().detach()

        self.dset.t_start = self.dset.his_t[0] if t_start is None else t_start
        self.dset.t_end = self.dset.his_t[-1] if t_end is None else t_end

        res = self.dset.mle()  # ← We'll convert this in next step
        print(res)

        # Update params from result
        g0_cov = np.array([[res.x[3], res.x[4]], [res.x[4], res.x[5]]])
        g2_cov = np.array([[res.x[6], res.x[7]], [res.x[7], res.x[8]]])
        self.dset = STHPDataset(
            s_mu=np.mean(train_s, axis=0),
            g0_cov=g0_cov,
            g2_cov=g2_cov,
            alpha=res.x[0],
            beta=res.x[1],
            mu=res.x[2]
        )

    def test(self, test_s, test_t, n):
        self.dset.his_s = torch.tensor(test_s, dtype=torch.float32, device=DEVICE)
        self.dset.his_t = torch.tensor(test_t, dtype=torch.float32, device=DEVICE)

        predict_s = []
        predict_t = []

        for i in tqdm(range(n, len(test_s))):
            s, t = self.dset.predict_next(i - 1)
            predict_s.append(s.cpu().numpy())
            predict_t.append(t)

        return np.vstack(predict_s), np.array(predict_t)

"""
Simulate Spatio-Temporal Hawkes Process
"""

class STHPDataset(SyntheticDataset):
    def __init__(self, s_mu, g0_cov, g2_cov, alpha, beta, mu, max_history=100, dist_only=False):
        super().__init__(dist_only)
        self.max_history = max_history
        self.alpha = alpha
        self.beta = beta
        self.mu = mu

        self.s_mu = torch.tensor(s_mu, dtype=torch.float32, device=DEVICE) if s_mu is not None else None
        self.g0_cov = torch.tensor(g0_cov, dtype=torch.float32, device=DEVICE)
        self.g2_cov = torch.tensor(g2_cov, dtype=torch.float32, device=DEVICE)

        self.g0_ic = torch.linalg.inv(self.g0_cov)
        self.g0_sidc = 1 / torch.sqrt(torch.linalg.det(self.g0_cov))
        self.g2_ic = torch.linalg.inv(self.g2_cov)
        self.g2_sidc = 1 / torch.sqrt(torch.linalg.det(self.g2_cov))

    def trunc(self, his):
        return his[-self.max_history:] if len(his) > self.max_history else his

    def lamb_st(self, s, t):
        s = torch.tensor(s, dtype=torch.float32, device=DEVICE) if not torch.is_tensor(s) else s
        t = torch.tensor(t, dtype=torch.float32, device=DEVICE)

        valid_idx = self.his_t < t
        his_t = self.trunc(self.his_t[valid_idx])
        his_s = self.trunc(self.his_s[valid_idx])

        if len(his_t) == 0:
            excitation = 0.0
        else:
            g1_vals = SyntheticDataset.g1(t, his_t, self.alpha, self.beta)
            g2_vals = SyntheticDataset.g2(s, his_s, self.g2_sidc, self.g2_ic)
            excitation = torch.sum(g1_vals * g2_vals)

        base = self.mu * SyntheticDataset.g0(s, self.s_mu, self.g0_sidc, self.g0_ic)
        return base + excitation

    def predict_next(self, i):
        raise NotImplementedError("Use a torch-based integrator to replace quad_vec and quad.")

    def generate_offsprings(self, t_i, s_i, verbose=False):
        t = t_i
        count = 0
        while True:
            m = self.alpha * np.exp(-self.beta * (t - t_i))
            t += np.random.exponential(scale=1 / m)
            if t > self.t_end:
                break
            lamb = self.alpha * np.exp(-self.beta * (t - t_i))
            if lamb / m >= np.random.uniform():
                s = np.random.multivariate_normal(s_i.squeeze(), self.g2_cov.cpu().numpy())
                s = np.expand_dims(s.astype("float32"), 0)
                n = len(self.his_t[self.his_t < t])
                self.his_s = np.insert(self.his_s, n, s, axis=0)
                self.his_t = np.insert(self.his_t, n, t)
                count += 1
        if verbose:
            print(f"{count} offsprings generated for event at {t_i}")

    def lamb_St(self, mu, his_s, t):
        if his_s.nelement() == 0:
            influence = 0
        else:
            influence = torch.sum(self.g2_mats[self.stoi(his_s)].reshape(-1, self.x_num * self.y_num), dim=0)
        val = mu * torch.exp(self.g0_mat * self.beta * t - self.alpha * influence)
        return torch.minimum(val, self.lamb_max * torch.ones_like(val))
    
    def generate(self, t_start, t_end, verbose=False):
        self.t_start = t_start
        self.t_end = t_end
        t = t_start
        self.his_s = np.zeros((0, 2))
        self.his_t = np.array([])

        count = 0
        while True:
            count += 1
            t += np.random.exponential(scale=1 / self.mu)
            if t > t_end:
                break
            s = np.random.multivariate_normal(self.s_mu.cpu().numpy(), self.g0_cov.cpu().numpy())
            s = np.expand_dims(s.astype("float32"), 0)
            self.his_s = np.vstack((self.his_s, s))
            self.his_t = np.append(self.his_t, t)
        if verbose:
            print(f"{count} 0-generation events generated")

        t = t_start
        n = 0
        while True:
            self.generate_offsprings(self.his_t[n], self.his_s[n], verbose)
            try:
                n = next(x[0] for x in enumerate(self.his_t) if x[1] > t)
                t = self.his_t[n]
            except StopIteration:
                break

    def nll(self, alpha, beta, mu, g0_cov, g2_cov):
        g0_ic = np.linalg.inv(g0_cov)
        g0_sidc = 1 / np.sqrt(np.linalg.det(g0_cov))
        g2_ic = np.linalg.inv(g2_cov)
        g2_sidc = 1 / np.sqrt(np.linalg.det(g2_cov))
        s_mu = np.mean(self.his_s, axis=0)

        term_1 = 0
        for i in range(1, len(self.his_s)):
            lamb = mu * self.g0(self.his_s[i], s_mu, g0_sidc, g0_ic) + \
                   np.sum(self.g1(self.his_t[i], self.trunc(self.his_t[:i]), alpha, beta) * \
                          self.g2(self.his_s[i], self.trunc(self.his_s[:i]), g2_sidc, g2_ic))
            term_1 -= np.log(lamb)

        term_2 = mu * (self.t_end - self.t_start)
        term_2 -= alpha / beta * np.sum((np.exp(-beta * (self.t_end - self.his_t)) - 1))

        return term_1 + term_2

    def mle(self):
        xinit = [2, 2, 2, 2, 0, 2, 2, 0, 2]
        bnds = [(0, None), (eps, None), (0, None), (eps, None), (0, None), (eps, None),
                (eps, None), (0, None), (eps, None)]
        cons = [
            {'type': 'ineq', 'fun': lambda x: x[3] * x[5] - x[4] * x[4]},
            {'type': 'ineq', 'fun': lambda x: x[6] * x[8] - x[7] * x[7]}
        ]
        obj_fun = lambda x: self.nll(x[0], x[1], x[2],
                                     np.array([[x[3], x[4]], [x[4], x[5]]]),
                                     np.array([[x[6], x[7]], [x[7], x[8]]]))
        return minimize(obj_fun, x0=xinit, bounds=bnds, constraints=cons)

    def plot_intensity(self, s=None, t_start=None, t_end=None, color='blue'):
        if s is None:
            s = self.s_mu.cpu().numpy()[np.newaxis, :]
        if t_start is None:
            t_start = self.t_start
        if t_end is None:
            t_end = self.t_end

        width, _ = plt.figaspect(.1)
        _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(width, width/2))

        x = np.arange(1001) / 1000
        x = eps + t_start + x * (t_end - t_start)
        y = [self.lamb_st(s, t).cpu().item() for t in x]
        ax1.plot(x, y, color, label=f'Intensity at ({s[0,0]}, {s[0,1]})')
        ax1.set_xlim([t_start, t_end])
        ax1.legend()

        idx = np.logical_and(self.his_t >= t_start, self.his_t < t_end)
        ax2.stem(self.his_t[idx], np.sqrt(np.sum(np.square(self.his_s[idx] - s), axis=1)),
                 use_line_collection=True, label=f'Events(height = dist to ({s[0,0]}, {s[0,1]}))')
        ax2.set_xlim([t_start, t_end])
        ax2.invert_yaxis()
        ax2.legend()
    
    def get_lamb_st(self, x_num, y_num, t_num, t_start, t_end):
        mask   = (self.his_t >= t_start) & (self.his_t < t_end)
        his_s0 = self.his_s[mask]
        his_t0 = self.his_t[mask]

        # build a flat grid of N = x_num*y_num points
        x = torch.linspace(his_s0[:,0].min(), his_s0[:,0].max(), x_num, device=DEVICE)
        y = torch.linspace(his_s0[:,1].min(), his_s0[:,1].max(), y_num, device=DEVICE)
        gx, gy = torch.meshgrid(x, y, indexing='ij')
        s_grids = torch.stack([gx.flatten(), gy.flatten()], dim=1)

        t_range = torch.linspace(t_start, t_end, t_num, device=DEVICE)
        mu_vec  = torch.full((x_num*y_num,), self.mu, device=DEVICE)

        lambs = []
        for t in tqdm(t_range):
            # history up to t
            sub_mask  = his_t0 < t
            lamb_flat = self.lamb_St(mu_vec, his_s0[sub_mask], t)
            # reshape and move to CPU/NumPy
            lambs.append(lamb_flat.view(x_num, y_num).T.cpu().numpy())

        return lambs, x.cpu().numpy(), y.cpu().numpy(), t_range.cpu().numpy()
"""
Simulate Discrete-Event Brownian Motion
"""
class DEBMDataset(SyntheticDataset):
    def __init__(self, s_mu, g2_cov, alpha, beta, mu, max_history=100, dist_only=False):
        super().__init__(dist_only)
        self.s_mu = torch.tensor(s_mu, dtype=torch.float32, device=DEVICE).clone().detach()
        self.g2_cov = torch.tensor(g2_cov, dtype=torch.float32, device=DEVICE).clone().detach()
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.max_history = max_history

        self.g2_ic = torch.linalg.inv(self.g2_cov)
        self.g2_sidc = 1 / torch.sqrt(torch.linalg.det(self.g2_cov))

    def lamb_t(self, t):
        t = torch.tensor(t, dtype=torch.float32, device=DEVICE).clone().detach()
        last_t = self.his_t[self.his_t < t][-1]
        return self.mu + self.alpha * torch.exp(-self.beta * (t - last_t))

    def lamb_st(self, s, t):
        s = torch.tensor(s, dtype=torch.float32, device=DEVICE).clone().detach() if not torch.is_tensor(s) else s
        last_s = self.his_s[self.his_t < t][-1:].to(DEVICE)
        return SyntheticDataset.g2(last_s, s, self.g2_sidc, self.g2_ic) * self.lamb_t(t)

    def generate(self, t_start, t_end, verbose=False):
        t = 0
        self.his_s = self.s_mu.view(1, 2).clone().detach()
        self.his_t = torch.tensor([-eps], dtype=torch.float32, device=DEVICE).clone().detach()

        while True:
            lamb_t = self.lamb_t(t)
            if self.beta >= 0:
                l = float('inf')
                m = self.lamb_t(t + eps)
            else:
                l = 2 / lamb_t
                m = self.lamb_t(t + l)

            delta_t = torch.distributions.Exponential(1 / m).sample().item()

            if t + delta_t > t_end:
                break
            if delta_t > l:
                t += l
                continue
            else:
                t += delta_t
                new_lamb_t = self.lamb_t(t)
                if new_lamb_t / m >= torch.rand(1, device=DEVICE).item():
                    if verbose:
                        print("----")
                        print(f"t:  {t}")
                        print(f"λt: {new_lamb_t.item()}")

                    cov_cpu = self.g2_cov.cpu().numpy()
                    loc_cpu = self.his_s[-1].cpu().numpy()
                    s = np.random.multivariate_normal(loc_cpu, cov_cpu)
                    s = torch.tensor(s, dtype=torch.float32, device=DEVICE).view(1, 2).clone().detach()
                    self.his_s = torch.cat((self.his_s, s), dim=0)
                    self.his_t = torch.cat((self.his_t, torch.tensor([t], dtype=torch.float32, device=DEVICE).clone().detach()))
