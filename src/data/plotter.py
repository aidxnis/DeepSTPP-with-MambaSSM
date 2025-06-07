import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_diff(outputs, targets, portion=1):
    """
    outputs: [batch, lookahead, 3] or [batch, 3]
    targets: same shape
    """
    out_t = torch.as_tensor(outputs, device=DEVICE)
    tgt_t = torch.as_tensor(targets, device=DEVICE)
    if out_t.ndim == 2:
        out_t = out_t.unsqueeze(1)
        tgt_t = tgt_t.unsqueeze(1)

    L = int(out_t.shape[0] * portion)
    out_t, tgt_t = out_t[:L], tgt_t[:L]

    out = out_t.cpu().numpy()
    tgt = tgt_t.cpu().numpy()

    n, lookahead = out.shape[0], out.shape[1]

    plt.figure(figsize=(14, 10), dpi=180)
    # Latitude
    plt.subplot(2, 2, 1)
    for i in range(lookahead):
        plt.plot(range(i, n), out[:n - i, i, 0], "-o", label=f"Pred {i}")
    plt.plot(tgt[:, 0, 0], "-o", color="b", label="Actual")
    plt.ylabel("Latitude")
    plt.legend()

    # Longitude
    plt.subplot(2, 2, 2)
    for i in range(lookahead):
        plt.plot(range(i, n), out[:n - i, i, 1], "-o", label=f"Pred {i}")
    plt.plot(tgt[:, 0, 1], "-o", color="b", label="Actual")
    plt.ylabel("Longitude")
    plt.legend()

    # Δt
    plt.subplot(2, 2, 3)
    for i in range(lookahead):
        plt.plot(range(i, n), out[:n - i, i, 2], "-o", label=f"Pred {i}")
    plt.plot(tgt[:, 0, 2], "-o", color="b", label="Actual")
    plt.ylabel("Δt (hours)")
    plt.legend()

    plt.savefig("result.png")
    plt.close()


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration},
    }


def inverse_transform(xg, yg, tg, scaler):
    """
    GPU tensors in, GPU tensors out. Expands x/y to match t before stacking.
    """
    max_len = max(len(xg), len(yg), len(tg))

    xg_pad = torch.full((max_len,), xg[-1], device=xg.device)
    xg_pad[:len(xg)] = xg

    yg_pad = torch.full((max_len,), yg[-1], device=yg.device)
    yg_pad[:len(yg)] = yg

    arr = torch.stack([xg_pad, yg_pad, tg], dim=1).cpu().numpy()
    inv = scaler.inverse_transform(arr)
    inv_t = torch.as_tensor(inv, device=xg.device)
    return inv_t[:, 0], inv_t[:, 1], inv_t[:, 2]



def plot_lambst_static(
    lambs, x_range, y_range, t_range, fps,
    scaler=None, cmin=None, cmax=None,
    history=None, decay=0.3, base_size=300,
    cmap="magma", fn="result.mp4"
):
    """
    lambs: list of torch.Tensor [H,W] on DEVICE
    x_range,y_range,t_range: 1D torch.Tensor on DEVICE
    """
    # optionally inverse‐scale
    if scaler is not None:
        x_range, y_range, t_range = inverse_transform(x_range, y_range, t_range, scaler)

    # compute color limits on GPU
    all_max = max(l.max() for l in lambs)
    cmin = 0 if cmin is None else cmin
    cmax = all_max.item() if (cmax is None or cmax == "outlier") else cmax
    cmid = cmin + 0.9 * (cmax - cmin)

    # build grid for plotting
    gx, gy = torch.meshgrid(x_range, y_range, indexing="ij")
    X, Y = gx.cpu().numpy(), gy.cpu().numpy()

    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = fig.add_subplot(
        111, projection="3d",
        xlabel="x", ylabel="y", zlabel="λ",
        zlim=(cmin, cmax),
        title="Spatio-temporal Conditional Intensity"
    )
    ax.title.set_position([0.5, 0.95])
    text = ax.text(
        x_range[0].item(), y_range[0].item(), cmax,
        f"t={t_range[0].item():.2f}", fontsize=10
    )

    # initial surface
    surf = ax.plot_surface(
        X, Y, lambs[0].cpu().numpy(),
        cmap=cmap, rstride=1, cstride=1
    )

    pts = None
    if history is not None:
        hs_t, ht_t = history
        # platform at cmid
        Z0 = np.full_like(X, cmid)
        ax.plot_surface(X, Y, Z0, color="white", alpha=0.2)
        pts = ax.scatter3D([], [], [], color="black")

    pbar = tqdm(total=len(t_range) + 2)
    def update(i):
        nonlocal surf, pts
        t = t_range[i].item()

        surf.remove()
        surf = ax.plot_surface(
            X, Y, lambs[i].cpu().numpy(),
            cmap=cmap, rstride=1, cstride=1
        )
        text.set_text(f"t={t:.2f}")

        if history is not None:
            hs_cpu = hs_t.cpu().numpy()
            ht_cpu = ht_t.cpu().numpy()
            mask = (ht_cpu <= t) & (ht_cpu >= t_range[0].item())
            locs, times = hs_cpu[mask], ht_cpu[mask]
            sizes = np.exp((times - t) * decay) * base_size
            zs = np.full_like(sizes, cmid)
            pts.remove()
            pts = ax.scatter3D(
                locs[:, 0], locs[:, 1], zs,
                c="black", s=sizes, marker="x"
            )
        pbar.update()

    ani = animation.FuncAnimation(
        fig, update, frames=len(t_range),
        interval=1000/fps
    )
    ani.save(fn, writer="ffmpeg", fps=fps)
    plt.close(fig)
    return ani


def plot_lambst_interactive(
    lambs, x_range, y_range, t_range,
    cmin=None, cmax=None, scaler=None, heatmap=False
):
    """
    Still uses CPU‐side NumPy for Plotly.
    """
    if scaler is not None:
        x_range, y_range, t_range = inverse_transform(x_range, y_range, t_range, scaler)

    xs = x_range.cpu().numpy()
    ys = y_range.cpu().numpy()
    ts = t_range.cpu().numpy()

    cmin = 0 if cmin is None else cmin
    if cmax == "outlier":
        cmax = max(l.cpu().numpy().max() for l in lambs)
    cmax = cmax or max(l.cpu().numpy().max() for l in lambs)

    frames = []
    for i, lt in enumerate(lambs):
        Z = lt.cpu().numpy()
        data = [go.Heatmap(z=Z, x=xs, y=ys, zmin=cmin, zmax=cmax)] if heatmap else [go.Surface(z=Z, x=xs, y=ys, cmin=cmin, cmax=cmax)]
        frames.append(go.Frame(data=data, name=f"{ts[i]:.2f}"))

    fig = go.Figure(frames=frames)
    # initial trace
    if heatmap:
        fig.add_trace(go.Heatmap(z=frames[0].data[0].z, x=xs, y=ys, zmin=cmin, zmax=cmax))
    else:
        fig.add_trace(go.Surface(z=frames[0].data[0].z, x=xs, y=ys, cmin=cmin, cmax=cmax))

    slider = {
        "pad": {"b": 10, "t": 60},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {"args": [[f.name], frame_args(0)], "label": f.name, "method": "animate"}
            for f in frames
        ]
    }
    fig.update_layout(
        title="Spatio-temporal Conditional Intensity",
        width=600, height=600,
        scene=dict(zaxis=dict(range=[cmin, cmax], autorange=False),
                   aspectratio=dict(x=1, y=1, z=1)),
        updatemenus=[{
            "buttons": [
                {"args": [None, frame_args(1)], "label": "►", "method": "animate"},
                {"args": [[None], frame_args(0)], "label": "❚❚", "method": "animate"}
            ],
            "direction": "left","pad":{"r":10,"t":70},"type":"buttons","x":0.1,"y":0
        }],
        sliders=[slider]
    )
    fig.show()


class TrajectoryPlotter:
    """
    GPU→CPU helper for Plotly trajectories.
    """
    def __init__(self):
        self.data = []
        self.layout = go.Layout(
            width=1200, height=600,
            scene=dict(
                camera=dict(up=dict(x=1, y=0, z=0),
                            eye=dict(x=0, y=2.5, z=0)),
                xaxis=dict(title="latitude"),
                yaxis=dict(title="longitude"),
                zaxis=dict(title="time"),
                aspectmode="manual", aspectratio=dict(x=1, y=1, z=3)
            ),
            showlegend=True
        )

    def compare(self, outputs, targets):
        out_t = torch.as_tensor(outputs, device=DEVICE)
        tgt_t = torch.as_tensor(targets, device=DEVICE)
        if out_t.ndim == 2:
            out_t = out_t.unsqueeze(1)
            tgt_t = tgt_t.unsqueeze(1)

        out = out_t.cpu().numpy()
        tgt = tgt_t.cpu().numpy()

        # actual
        tt = np.append(0, np.cumsum(tgt[:, 0, 2]))
        self.add_trace(tgt[:, 0, 0], tgt[:, 0, 1], tt, name="actual")

        n, la = out.shape[0], out.shape[1]
        for i in range(la):
            ot = np.append(0, np.append(0, np.cumsum(tgt[:n-i-1, 0, 2])) + out[:n-i, i, 2])
            self.add_trace(out[:n-i, i, 0], out[:n-i, i, 1], ot, name=f"pred {i}")

    def add_trace(self, x, y, z, name=None, color=None):
        self.data.append(go.Scatter3d(
            x=x, y=y, z=z, name=name,
            mode="lines+markers",
            marker=dict(size=4, symbol="circle", color=color),
            line=dict(width=3, color=color),
            opacity=0.6
        ))

    def show(self):
        fig = go.Figure(data=self.data, layout=self.layout)
        fig.show()
