# Based on Jisung Byun's work, modified for PyTorch


import torch
import torch.nn.functional as F

class Coolant_Evaporator(object):
    def __init__(self, coefficients):
        # Refrigerant property fitting coefficients
        self.coefficients = coefficients
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.coeff_Cp = torch.tensor(self.coefficients["Cp"], dtype=torch.float32).to(self.device)
        self.coeff_D = torch.tensor(self.coefficients["D"], dtype=torch.float32).to(self.device)
        self.coeff_mu = torch.tensor(self.coefficients["mu"].reshape(-1), dtype=torch.float32).to(self.device)
        self.coeff_k = torch.tensor(self.coefficients["k"], dtype=torch.float32).to(self.device)

    def poly1(self, T): return torch.cat([T, torch.ones((len(T), 1), device=T.device)], dim=-1)
    def poly2(self, T): return torch.cat([T**2, T, torch.ones((len(T), 1), device=T.device)], dim=-1)

    # Normalization function for temperature fitting
    def norm_T(self, T):
        return (T - 45) / 48.3477
    
    # Polynomial fit for Cp
    def Cp(self, T):
        Cp = torch.bmm(self.coeff_Cp.unsqueeze(0).repeat(len(T), 1, 1), self.poly1(T).unsqueeze(-1))
        return Cp.squeeze(-1)

    # Polynomial fit for D
    def D(self, T):
        D = torch.bmm(self.coeff_D.unsqueeze(0).repeat(len(T),1,1), self.poly2(T).unsqueeze(-1))
        return D.squeeze(-1)

    # Exponential fit for mu
    def mu(self, T):
        norm_T = self.norm_T(T)
        mu = self.coeff_mu[0] * torch.exp(-self.coeff_mu[1] * norm_T)
        return mu

    # Polynomial fit for k
    def k(self, T):
        k = torch.bmm(self.coeff_k.unsqueeze(0).repeat(len(T),1,1), self.poly2(T).unsqueeze(-1))
        return k.unsqueeze(-1)

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