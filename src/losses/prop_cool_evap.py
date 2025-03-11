import casadi as ca


class Coolant_evaporator(object):
    def __init__(self, coefficients):
        x = ca.SX.sym("x", 1)
        
        # Refrigerant property fitting coefficients
        self.coefficients = coefficients
        
        self.coeff_Cp = self.coefficients["Cp"]
        self.coeff_D = self.coefficients["D"]
        self.coeff_mu = self.coefficients["mu"].reshape(-1)
        self.coeff_k = self.coefficients["k"]
        
        # Arbitrary vectors for fitting
        self.poly1 = ca.Function("poly1", [x], [ca.vcat([x, 1])])
        self.poly2 = ca.Function("poly2", [x], [ca.vcat([x**2, x, 1])])
        
        # Normalization functions for fittinig
        self.norm_T = ca.Function("norm_T", [x], [(x - 45) / 48.3477])
        
        
    # Properties
    def Cp(self, T):
        Cp = self.coeff_Cp @ self.poly1(T)
        return Cp
    
    def D(self, T):
        D = self.coeff_D @ self.poly2(T)
        return D
    
    def mu(self, T):
        mu = self.coeff_mu[0] * ca.exp(-self.coeff_mu[1] * self.norm_T(T))
        return mu
    
    def k(self, T):
        k = self.coeff_k @ self.poly2(self.norm_T(T))
        return k