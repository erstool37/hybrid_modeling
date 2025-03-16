import torch
import torch.nn.functional as F

class Coolant_Evaporator(object):
    def __init__(self, coefficients):
        # Refrigerant property fitting coefficients
        self.coefficients = coefficients
        
        self.coeff_Cp = torch.tensor(self.coefficients["Cp"], dtype=torch.float32)
        self.coeff_D = torch.tensor(self.coefficients["D"], dtype=torch.float32)
        self.coeff_mu = torch.tensor(self.coefficients["mu"].reshape(-1), dtype=torch.float32)
        self.coeff_k = torch.tensor(self.coefficients["k"], dtype=torch.float32)

    # Normalization function for temperature fitting
    def norm_T(self, T):
        return (T - 45) / 48.3477
    
    # Polynomial fit for Cp
    def Cp(self, T):
        # poly1(T) = [T, 1]
        poly1 = torch.stack([T, torch.tensor([1.0], device=T.device)], dim=-1)
        Cp = torch.matmul(self.coeff_Cp, poly1.T).squeeze(1)
        return Cp
    
    # Polynomial fit for D
    def D(self, T):
        poly2 = torch.stack([T**2, T, torch.tensor([1.0], device=T.device)], dim=-1)
        D = torch.matmul(self.coeff_D, poly2.T).squeeze(1)
        return D
    
    # Exponential fit for mu
    def mu(self, T):
        norm_T = self.norm_T(T)
        mu = self.coeff_mu[0] * torch.exp(-self.coeff_mu[1] * norm_T).squeeze(1)
        print("mu", mu)
        return mu
    
    # Polynomial fit for k
    def k(self, T):
        norm_T = self.norm_T(T)
        poly2 = torch.stack([norm_T**2, norm_T, torch.tensor([1.0], device=T.device)], dim=-1)
        k = torch.matmul(self.coeff_k, poly2.T).squeeze(1)
        return k


"""
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
"""