import numpy as np
from scipy.optimize import minimize
from collections import deque
import torch
from utils import Xdenormalizer, Udenormalizer, Xnormalizer, Unormalizer

def mpc(model_input, x_sp_t, u_sp_t,
        f_p, f_x, P, Q, R, descaler,
        u_min, u_max, x_min, x_max):
    # last 30‐step normalized horizon
    x_seq_norm = model_input[0, :, :2].cpu().numpy().astype(np.float64)   # (30,2)
    u_seq_norm = model_input[0, :, 2:7].cpu().numpy().astype(np.float64)  # (30,5)

    # unnormalize the last state & control
    x0_norm      = x_seq_norm[-1]
    u0_norm_full = u_seq_norm[-1]
    x0_real      = Xdenormalizer(
                       torch.tensor(x0_norm, dtype=torch.float32),
                       descaler, "optim"
                   ).cpu().numpy()
    u0_real_full = Udenormalizer(
                       torch.tensor(u0_norm_full, dtype=torch.float32),
                       descaler, "optim"
                   ).cpu().numpy()

    
    m_cool_fixed = float(u0_real_full[3])
    T_cool_fixed = float(u0_real_full[4])

    x_sp_np = x_sp_t.cpu().numpy().astype(np.float64)
    u_sp_np = u_sp_t.cpu().numpy().astype(np.float64)

    N = 5
    z0 = np.tile([u0_real_full[0], u0_real_full[2]], N).astype(np.float64)

    # 4) build normalized‐history deques
    xh_norm = deque(x_seq_norm.tolist(), maxlen=30)
    uh_norm = deque(u_seq_norm.tolist(), maxlen=30)

    # 5) shooting cost
    def cost(z):
        x_real = x0_real.copy()
        xh_n   = deque(xh_norm, maxlen=30)
        uh_n   = deque(uh_norm, maxlen=30)
        total  = 0.0
        for k in range(N):
            # build full control in real units
            u1, u3 = float(z[2*k]), float(z[2*k+1])
            uk = np.array([u1, u1, u3, m_cool_fixed, T_cool_fixed],
                          dtype=np.float32)

            # append real→normalized to histories
            x_norm = Xnormalizer(
                         torch.tensor(x_real, dtype=torch.float32),
                         descaler, "optim"
                     ).cpu().numpy()
            u_norm = Unormalizer(
                         torch.tensor(uk, dtype=torch.float32),
                         descaler, "optim"
                     ).cpu().numpy()
            xh_n.append(x_norm.tolist())
            uh_n.append(u_norm.tolist())

            # predict p_k from normalized histories
            p_k = f_p(np.array(xh_n), np.array(uh_n))
            print(f"p_k: {p_k}")

            # one‐step dynamics in real units
            x_t = f_x(
                torch.tensor(x_real, dtype=torch.float32),
                torch.tensor(uk,     dtype=torch.float32),
                torch.tensor(p_k,    dtype=torch.float32)
            )
            x_real = x_t.detach().cpu().numpy()

            # stage cost (tracking)
            dx = x_real - x_sp_np
            du = uk     - u_sp_np
            total += dx @ Q @ dx + du @ R @ du

        # terminal cost
        dxN = x_real - x_sp_np
        total += dxN @ P @ dxN
        return float(total)

    # 6) bounds on decision vars z = [m_in,h_in] × 5
    bounds = []
    for _ in range(N):
        bounds += [
            (float(u_min[0]), float(u_max[0])),  # m_ref_in
            (float(u_min[2]), float(u_max[2]))   # h_ref_in
        ]

    # 7) solve with SLSQP in float64
    res = minimize(
        cost, z0, bounds=bounds,
        method='SLSQP',
        options={'maxiter':1000}
    )
    z_opt = res.x

    # 8) rebuild full optimal u‐horizon in real units
    U_opt = np.zeros((N,5), dtype=np.float32)
    for k in range(N):
        u1, u3 = float(z_opt[2*k]), float(z_opt[2*k+1])
        U_opt[k] = [u1, u1, u3, m_cool_fixed, T_cool_fixed]

    # 9) one‐step‐ahead prediction x1
    xh_n = deque(xh_norm, maxlen=30)
    uh_n = deque(uh_norm, maxlen=30)
    x1_real = x0_real.copy()
    uk0 = U_opt[0]
    # append for p0 calc
    x_norm0 = Xnormalizer(
                  torch.tensor(x0_real, dtype=torch.float32),
                  descaler, "optim"
              ).cpu().numpy()
    u_norm0 = Unormalizer(
                  torch.tensor(uk0, dtype=torch.float32),
                  descaler, "optim"
              ).cpu().numpy()
    xh_n.append(x_norm0.tolist())
    uh_n.append(u_norm0.tolist())
    p0 = f_p(np.array(xh_n), np.array(uh_n))
    x1_t = f_x(
        torch.tensor(x0_real, dtype=torch.float32),
        torch.tensor(uk0,     dtype=torch.float32),
        torch.tensor(p0,      dtype=torch.float32)
    )
    x1_pred = x1_t.detach()

    # 10) return optimized u0 and x1 prediction
    u0_opt = torch.tensor(U_opt[0], dtype=torch.float32)
    return u0_opt, x1_pred