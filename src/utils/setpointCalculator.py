from scipy.optimize import minimize
import numpy as np

def setpointCalculator(function, x, u, p, descaler, Cp_cool, T_cool_target, tol=1e-4, method='SLSQP',):
    # x, u, p initialization
    x = x[:,0,:].squeeze(0).detach().cpu().numpy()
    u = u[:,0,:].squeeze(0).detach().cpu().numpy()
    p = p[:,0,:].squeeze(0).detach().cpu().numpy()
    x0 = x
    u0 = u
    x0 = np.concatenate((x0, u0), axis=0)

    def objective(var, p):
        x = var[:2]
        u = var[2:]
        xdot = function(x, u, p)
        return np.sum(xdot**2)
    
    # Bounds
    x_min = [110, 260]
    x_max = [225, 360]
    u_min = [0.01, 0.01, 200, 0.175, -13]
    u_max = [0.03, 0.03, 240, 0.5, 0]
    bounds = list(zip(x_min + u_min, x_max + u_max))

    # Constraints
    args = (Cp_cool, T_cool_target)
    constraints = {"type": "eq", "fun": T_cool_out, "args": args}

    result = minimize(objective, x0, args=(p,), bounds=bounds, constraints=constraints, tol=tol, method=method)
    
    return result.x, result.u

def T_cool_out(var, Cp_cool, T_cool_target):
    x = var[:2]
    u = var[2:]

    _, h_ref_out = x
    m_ref_in, _, h_ref_in, m_cool, T_cool_in = u

    Q_ref = m_ref_in * (h_ref_out - h_ref_in)
    Q_cool = m_cool * Cp_cool * (T_cool_target - T_cool_in)

    return Q_ref - Q_cool