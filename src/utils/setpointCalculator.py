import numpy as np
from scipy.optimize import minimize
import torch
from utils import Pdenormalizer, Xdenormalizer, Udenormalizer, Xnormalizer, Unormalizer

def predict_p(xu, model, descaler):
    model.eval()
    with torch.no_grad():
        xu_tensor = torch.tensor(xu, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 7)
        
        x = xu_tensor[:, :, :2].squeeze(0).squeeze(0)
        u = xu_tensor[:, :, 2:].squeeze(0).squeeze(0)
        
        norm_x = Xnormalizer(x, descaler, "optim")
        norm_u = Unormalizer(u, descaler, "optim")

        xu_norm = torch.cat([norm_x, norm_u], dim=-1)
        xu_horizon = xu_norm.repeat(1,30,1)
        p_scaled = model(xu_horizon).squeeze(0).squeeze(0)
        p_scaled = p_scaled[-1, :]
        p = Pdenormalizer(p_scaled, descaler, "optim").cpu().numpy()
    return p

def objective(var, model, descaler, function, fixed_u):
    x = var[:2]
    u_partial = var[2:]  # [u[0], u[2]]

    u_full = fixed_u.copy()
    u_full[0] = u_partial[0]
    u_full[1] = u_partial[1]
    u_full[2] = u_partial[2]

    xu = np.concatenate((x, u_full), axis=0)
    p = predict_p(xu, model, descaler)
    xdot = function(x, u_full, p)
    return np.sum(xdot**2)


def T_cool_out(var, Cp_cool, T_cool_target, fixed_u):
    x = var[:2]
    u_partial = var[2:]

    u_full = fixed_u.copy()
    u_full[0] = u_partial[0]
    u_full[1] = u_partial[1]
    u_full[2] = u_partial[2]

    _, h_ref_out = x
    m_ref_in, _, h_ref_in, m_cool, T_cool_in = u_full

    Q_ref = m_ref_in * (h_ref_out - h_ref_in)
    Q_cool = m_cool * Cp_cool * (T_cool_target - T_cool_in)
    dQ = np.array([Q_ref - Q_cool]).flatten()
    return dQ

def constraint_u(var):
    return np.array([var[2] - var[3]])

def setpointCalculator(function, x_tensor, u_tensor, model, descaler, Cp_cool, T_cool_target, tol=1e-4, method='SLSQP'):
    # Convert tensors to NumPy
    x = Xdenormalizer(x_tensor, descaler, "optim")
    u = Udenormalizer(u_tensor, descaler, "optim")
    x = x.detach().cpu().numpy()
    u = u.detach().cpu().numpy()
    fixed_u = u.copy()

    # Initial guess: x + selected u
    x0 = np.concatenate((x, [u[0], u[1], u[2]]))  # Only u[0], u[2] optimized

    # Bounds
    x_min = [110, 260]
    x_max = [225, 360]
    u_partial_min = [0.01, 0.01, 200]
    u_partial_max = [0.03, 0.03, 240]
    bounds = list(zip(x_min + u_partial_min, x_max + u_partial_max))

    # Constraint: outlet temp balance
    constraints = [{
        "type": "eq",
        "fun": T_cool_out,
        "args": (Cp_cool, T_cool_target, fixed_u)
    }, {
        "type": "eq",
        "fun": constraint_u
    }]

    # Optimization
    result = minimize(
        objective,
        x0,
        args=(model, descaler, function, fixed_u),
        bounds=bounds,
        constraints=constraints,
        tol=tol,
        method=method
    )

    # Rebuild full u
    x_opt = result.x[:2]
    u_partial_opt = result.x[2:]
    u_opt = fixed_u.copy()
    u_opt[0] = u_partial_opt[0]
    u_opt[1] = u_partial_opt[1]
    u_opt[2] = u_partial_opt[2]

    return x_opt, u_opt