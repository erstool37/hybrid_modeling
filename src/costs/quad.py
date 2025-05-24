import casadi as ca

def make_mpc_solver(f_step, nx, nu, Np, Nc, Q, R, P):
    x_min = ca.DM([100.0, 270.0])
    x_max = ca.DM([360.0, 380.0])
    u_min = ca.DM([0.01, 0.01, 163.0, 0.0001, -20.0])
    u_max = ca.DM([0.05, 0.05, 365.0, 1.0, 10.0])

    # Decision variables
    X = ca.MX.sym('X', nx, Np+1)
    U = ca.MX.sym('U', nu, Nc)

    # Parameters
    X0  = ca.MX.sym('X0', nx)
    Xsp = ca.MX.sym('Xsp', nx)
    Usp = ca.MX.sym('Usp', nu)

    g = []
    J = 0
    # Initial state constraint
    g.append(X[:, 0] - X0)

    # Build dynamics and stage cost
    for k in range(Nc):
        xk = X[:, k]
        uk = U[:, k]
        x_next = f_step(xk, uk)
        g.append(X[:, k+1] - x_next)
        dx = xk - Xsp
        du = uk - Usp
        J += ca.dot(dx, Q @ dx) + ca.dot(du, R @ du)

    # Terminal cost
    dxN = X[:, Np] - Xsp
    J += ca.dot(dxN, P @ dxN)

    # Pack decision variables and constraints
    w = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    g = ca.vertcat(*g)
    nlp = {'x': w, 'f': J, 'g': g}
    solver = ca.nlpsol('solver', 'ipopt', nlp)

    # Build box bounds for states and controls
    lbx, ubx = [], []
    for _ in range(Np+1):
        lbx += list(x_min)
        ubx += list(x_max)
    for _ in range(Nc):
        lbx += list(u_min)
        ubx += list(u_max)

    # Equality constraints all zero
    lbg = [0] * ((Nc + 1) * nx)
    ubg = [0] * ((Nc + 1) * nx)

    return solver, lbx, ubx, lbg, ubg, X0, Xsp, Usp