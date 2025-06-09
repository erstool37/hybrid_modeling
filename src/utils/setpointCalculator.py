import casadi as ca
import numpy as np
import torch
from utils import Xdenormalizer, Udenormalizer, Xnormalizer, Unormalizer, Pdenormalizer, Odenormalizer


def setpointCalculator(system_dynamics, go_step, x_tensor, u_tensor, others_tensor,
    model, scaler, descaler, Cp_cool, T_cool_target, tol=1e-4):

    x0_np = Xdenormalizer(x_tensor, descaler, "optim").cpu().numpy()
    u0_np = Udenormalizer(u_tensor, descaler, "optim").cpu().numpy()
    O_np = Odenormalizer(others_tensor, descaler, "optim").cpu().numpy()

    T_ref_in = float(O_np[0])
    T_cool_in = float(O_np[1])
    m_cool = float(O_np[2]) 
    Cp_cool = float(Cp_cool)
    fixed_u = u0_np.copy()
    
    # Variable stacking and initial guess
    z0 = np.hstack([300, 300, 0.02, 0.02, 0.8])
    # z0 = np.hstack([ x0_np, u0_np[:3] ])
    z = ca.SX.sym("z", 5)
    x_var   = z[0:2]        
    u_part  = z[2:5]

    u_full = ca.vertcat(u_part[0], u_part[1], fixed_u[2], u_part[2], fixed_u[4])
    # u_full = ca.vertcat(u_part[0], u_part[1], u_part[2], fixed_u[3], fixed_u[4])

    def predict_p_np(zval):
        xu = np.hstack([ zval[0:2], zval[2:5], fixed_u[3:], ])
        with torch.no_grad():
            xu_t = torch.tensor(xu, dtype=torch.float32)
            x_t  = xu_t[:2];  u_t = xu_t[2:]
            nx   = Xnormalizer(x_t, scaler, "optim")
            nu   = Unormalizer(u_t, scaler, "optim")
            inp  = torch.cat([nx,nu])
            hor  = inp.repeat(30,1).unsqueeze(0)
            ps   = model(hor).squeeze().detach()
            ps   = ps[-1,:]
            return Pdenormalizer(ps, descaler, "optim").cpu().numpy()
    p0 = predict_p_np(z0)
    eps_tp = float(p0[2])

    # Objective
    xdot = system_dynamics(x_var, u_full, p0)
    constraint = ca.dot(xdot, xdot)
    # f_sym = ca.dot(xdot, xdot)

    # constraints
    
    T_cool_sys = u_full[0] * (x_var[1] - u_full[2]) / (u_full[3] * float(Cp_cool)) + u_full[4]
    dQ = T_cool_sys - T_cool_target
    
    f_sym = ca.dot(dQ, dQ)
    
    # Q_max = 0.2 * u_full[3] * Cp_cool * (T_ref_in - T_cool_in) # in negative direction
    # Q_cool= u_full[3] * Cp_cool * (T_cool_target - u_full[4]) # m_cool * Cp_cool * (T_cool_target - T_cool_in)
    # dQ = Q_max - Q_cool

    eq1 = constraint
    # eq1 = dQ
    eq2 = u_part[0] - u_part[1]
    g_sym = ca.vertcat(eq1, eq2)

    # bounds
    lbz = [ 100, 200, 0.005, 0.005, 200 ]
    ubz = [ 400, 400, 0.05,  0.05,  300 ]

    # solve
    nlp = {"x": z, "f": f_sym, "g": g_sym}
    opts = {"ipopt.print_level": 5, "print_time" : True, "ipopt.tol": tol, "ipopt.max_iter" : 100000, "ipopt.output_file" : "ipopt.log"}
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    sol = solver(x0 = z0, lbx = lbz, ubx = ubz, lbg = [-1,0], ubg = [1,0])

    z_opt = sol["x"].full().flatten()
    x_opt = z_opt[0:2]
    u_opt = fixed_u.copy()
    u_opt[:3] = z_opt[2:5]

    return x_opt, u_opt