# Based on Jisung Byun's work, modified for PyTorch

import torch
import torch.nn as nn

class Refrigerant(nn.Module):
    def __init__(self, coefficients):
        super(Refrigerant, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # coefficients
        self.coeff_Tsat = torch.tensor(coefficients["Tsat"]).reshape(-1).float().to(self.device)

        self.coeff_vap_hsat = torch.tensor(coefficients["vap_hsat"]).float().to(self.device)
        self.coeff_vap_Dsat = torch.tensor(coefficients["vap_Dsat"]).float().to(self.device)
        self.coeff_vap_musat = torch.tensor(coefficients["vap_musat"]).float().to(self.device)
        self.coeff_vap_ksat = torch.tensor(coefficients["vap_ksat"]).float().to(self.device)
        self.coeff_vap_Cpsat = torch.tensor(coefficients["vap_Cpsat"]).float().to(self.device)
        self.coeff_vap_Prsat = torch.tensor(coefficients["vap_Prsat"]).float().to(self.device)
        self.coeff_vap_dhsatdp = torch.tensor(coefficients["vap_dhsatdp"]).float().to(self.device)
        self.coeff_vap_dDsatdp = torch.tensor(coefficients["vap_dDsatdp"]).float().to(self.device)

        self.coeff_liq_hsat = torch.tensor(coefficients["liq_hsat"]).float().to(self.device)
        self.coeff_liq_Dsat = torch.tensor(coefficients["liq_Dsat"]).float().to(self.device)
        self.coeff_liq_musat = torch.tensor(coefficients["liq_musat"]).float().to(self.device)
        self.coeff_liq_ksat = torch.tensor(coefficients["liq_ksat"]).float().to(self.device)
        self.coeff_liq_Cpsat = torch.tensor(coefficients["liq_Cpsat"]).float().to(self.device)
        self.coeff_liq_Prsat = torch.tensor(coefficients["liq_Prsat"]).float().to(self.device)
        self.coeff_liq_dhsatdp = torch.tensor(coefficients["liq_dhsatdp"]).float().to(self.device)
        self.coeff_liq_dDsatdp = torch.tensor(coefficients["liq_dDsatdp"]).float().to(self.device)

        self.coeff_vap_sph = torch.tensor(coefficients["vap_sph"]).float().to(self.device)
        self.coeff_vap_hps = torch.tensor(coefficients["vap_hps"]).float().to(self.device)
        self.coeff_vap_Dph = torch.tensor(coefficients["vap_Dph"]).float().to(self.device)
        self.coeff_vap_muph = torch.tensor(coefficients["vap_muph"]).float().to(self.device)
        self.coeff_vap_kph = torch.tensor(coefficients["vap_kph"]).float().to(self.device)
        self.coeff_vap_Cph = torch.tensor(coefficients["vap_Cph"]).float().to(self.device)
        self.coeff_vap_Tph = torch.tensor(coefficients["vap_Tph"]).float().to(self.device)
        self.coeff_vap_CpT = torch.tensor(coefficients["vap_CpT"]).float().to(self.device)
        self.coeff_vap_Prph = torch.tensor(coefficients["vap_Prph"]).float().to(self.device)
        self.coeff_vap_dsdp = torch.tensor(coefficients["vap_dsdp"]).float().to(self.device)
        self.coeff_vap_dsdh = torch.tensor(coefficients["vap_dsdh"]).float().to(self.device)
        self.coeff_vap_dhdp = torch.tensor(coefficients["vap_dhdp"]).float().to(self.device)
        self.coeff_vap_dhds = torch.tensor(coefficients["vap_dhds"]).float().to(self.device)
        self.coeff_vap_dDdp = torch.tensor(coefficients["vap_dDdp"]).float().to(self.device)
        self.coeff_vap_dDdh = torch.tensor(coefficients["vap_dDdh"]).float().to(self.device)

        self.coeff_liq_Dph = torch.tensor(coefficients["liq_Dph"]).float().to(self.device)
        self.coeff_liq_muph = torch.tensor(coefficients["liq_muph"]).float().to(self.device)
        self.coeff_liq_kph = torch.tensor(coefficients["liq_kph"]).float().to(self.device)
        self.coeff_liq_Cph = torch.tensor(coefficients["liq_Cph"]).float().to(self.device)
        self.coeff_liq_Tph = torch.tensor(coefficients["liq_Tph"]).float().to(self.device)
        self.coeff_liq_CpT = torch.tensor(coefficients["liq_CpT"]).float().to(self.device)
        self.coeff_liq_Prph = torch.tensor(coefficients["liq_Prph"]).float().to(self.device)
        self.coeff_liq_dDdp = torch.tensor(coefficients["liq_dDdp"]).float().to(self.device)
        self.coeff_liq_dDdh = torch.tensor(coefficients["liq_dDdh"]).float().to(self.device)

    # Normalization functions
    def norm_psat(self, x):
        return (x - 1670.0) / 955.3279

    def norm_Tsat(self, x):
        return (x - 52.5643) / 33.3173

    def norm_vap_p(self, x): return (x - 1300.9) / 939.1761
    def norm_vap_h(self, x): return (x - 432.5771) / 32.0814
    def norm_vap_T(self, x): return (x - 83.8534) / 34.6682
    def norm_vap_s(self, x): return (x - 1.7544) / 0.1025
    def norm_liq_p(self, x): return (x - 1984.0) / 880.9612
    def norm_liq_h(self, x): return (x - 212.5470) / 49.7295
    def norm_liq_T(self, x): return (x - 7.2744) / 36.7646
    # def log_vec(self, x): return torch.cat([torch.log(x), torch.ones((len(x), 1))], dim=-1)

    def poly7(self, x): return torch.cat([x**7, x**6, x**5, x**4, x**3, x**2, x, torch.ones((len(x), 1),device=x.device)], dim=-1)
    def poly8(self, x): return torch.cat([x**8, x**7, x**6, x**5, x**4, x**3, x**2, x, torch.ones((len(x), 1),device=x.device)], dim=-1)
    def poly9(self, x): return torch.cat([x**9, x**8, x**7, x**6, x**5, x**4, x**3, x**2, x,torch.ones((len(x), 1),device=x.device)], dim=-1)
    def poly14(self, x, y): return torch.cat([torch.ones((len(x), 1), device=x.device), x, y, x*y, y**2, x*(y**2), y**3, x*(y**3), y**4], dim=-1)
    def poly15(self, x, y): return torch.cat([torch.ones((len(x), 1), device=x.device), x, y, x*y, y**2, x*(y**2), y**3, x*(y**3), y**4, x*(y**4), y**5], dim=-1)
    def poly23(self, x, y): return torch.cat([torch.ones((len(x), 1), device=x.device), x, y, x**2, x*y, y**2, x**2*y, x*y**2, y**3], dim=-1)
    def poly24(self, x, y): return torch.cat([torch.ones((len(x), 1), device=x.device), x, y, x**2, x*y, y**2, x**2*y, x*y**2, y**3, x**2*y**2, x*y**3, y**4], dim=-1)
    def poly25(self, x, y): return torch.cat([torch.ones((len(x), 1), device=x.device), x, y, x**2, x*y, y**2, x**2*y, x*y**2, y**3, x**2*y**2, x*y**3, y**4, x**2*y**3, x*y**4, y**5], dim=-1)
    def poly45(self, x, y): return torch.cat([torch.ones((len(x), 1), device=x.device), x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3, x**4, x**3*y, x**2*y**2, x*y**3, y**4, x**4*y, x**3*y**2, x**2*y**3, x*y**4, y**5], dim=-1)
    def poly52(self, x, y): return torch.cat([torch.ones((len(x), 1), device=x.device), x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, x**4, x**3*y, x**2*y**2, x**5, x**4*y, x**3*y**2], dim=-1)
    def poly54(self, x, y): return torch.cat([torch.ones((len(x), 1), device=x.device), x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3, x**4, x**3*y, x**2*y**2, x*y**3, y**4, x**5, x**4*y, x**3*y**2, x**2*y**3, x*y**4], dim=-1)
    def poly55(self, x, y): return torch.cat([torch.ones((len(x), 1), device=x.device), x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3, x**4, x**3*y, x**2*y**2, x*y**3, y**4, x**5, x**4*y, x**3*y**2, x**2*y**3, x*y**4, y**5], dim=-1)

    # Saturation properties
    def Tsat(self, p): # Caution, log used, wrong gradients might lead to NaN
        a, b, c = self.coeff_Tsat
        Tsat = b / (a - torch.log10(p)) - c
        return Tsat
    
    def Psat(self, T):
        a, b, c = self.coeff_Tsat
        log_p = a - b / (T + c)
        return 10 ** log_p

    def vap_hsat(self, p):
        vap_hsat = torch.bmm(self.coeff_vap_hsat.unsqueeze(0).repeat(len(p),1,1), self.poly8(self.norm_psat(p)).unsqueeze(-1))
        return vap_hsat.squeeze(-1)

    def vap_Dsat(self, p):
        vap_Dsat = torch.bmm(self.coeff_vap_Dsat.unsqueeze(0).repeat(len(p),1,1), self.poly9(self.norm_psat(p)).unsqueeze(-1))
        return vap_Dsat.squeeze(-1)

    def vap_musat(self, p):
        vap_musat = torch.bmm(self.coeff_vap_musat.unsqueeze(0).repeat(len(p),1,1), self.poly9(self.norm_psat(p)).unsqueeze(-1))
        return vap_musat.squeeze(-1)

    def vap_ksat(self, p):
        vap_ksat = torch.bmm(self.coeff_vap_ksat.unsqueeze(0).repeat(len(p),1,1), self.poly9(self.norm_psat(p)).unsqueeze(-1))
        return vap_ksat.squeeze(-1)

    def vap_Cpsat(self, p):
        vap_Cpsat = torch.bmm(self.coeff_vap_Cpsat.unsqueeze(0).repeat(len(p),1,1), self.poly9(self.norm_psat(p)).unsqueeze(-1))
        return vap_Cpsat.squeeze(-1)

    def vap_Prsat(self, p):
        vap_Prsat = torch.bmm(self.coeff_vap_Prsat.unsqueeze(0).repeat(len(p),1,1), self.poly9(self.norm_psat(p)).unsqueeze(-1))
        return vap_Prsat.squeeze(-1)

    def vap_dhsatdp(self, p):
        vap_dhsatdp = torch.bmm(self.coeff_vap_dhsatdp.unsqueeze(0).repeat(len(p),1,1), self.poly7(self.norm_psat(p)).unsqueeze(-1))
        return vap_dhsatdp.squeeze(-1)

    def vap_dDsatdp(self, p):
        vap_dDsatdp = torch.bmm(self.coeff_vap_dDsatdp.unsqueeze(0).repeat(len(p),1,1), self.poly8(self.norm_psat(p)).unsqueeze(-1))
        return vap_dDsatdp.squeeze(-1)

    def liq_hsat(self, p):
        liq_hsat = torch.bmm(self.coeff_liq_hsat.unsqueeze(0).repeat(len(p),1,1), self.poly9(self.norm_psat(p)).unsqueeze(-1))
        return liq_hsat.squeeze(-1)

    def liq_Dsat(self, p):
        liq_Dsat = torch.bmm(self.coeff_liq_Dsat.unsqueeze(0).repeat(len(p),1,1), self.poly9(self.norm_psat(p)).unsqueeze(-1))
        return liq_Dsat.squeeze(-1)

    def liq_musat(self, p):
        liq_musat = torch.bmm(self.coeff_liq_musat.unsqueeze(0).repeat(len(p),1,1), self.poly9(self.norm_psat(p)).unsqueeze(-1))
        return liq_musat.squeeze(-1)

    def liq_ksat(self, p):
        liq_ksat = torch.bmm(self.coeff_liq_ksat.unsqueeze(0).repeat(len(p),1,1), self.poly8(self.norm_psat(p)).unsqueeze(-1))
        return liq_ksat.squeeze(-1)

    def liq_Cpsat(self, p):
        liq_Cpsat = torch.bmm(self.coeff_liq_Cpsat.unsqueeze(0).repeat(len(p),1,1), self.poly9(self.norm_psat(p)).unsqueeze(-1))
        return liq_Cpsat.squeeze(-1)

    def liq_Prsat(self, p):
        liq_Prsat = torch.bmm(self.coeff_liq_Prsat.unsqueeze(0).repeat(len(p),1,1), self.poly8(self.norm_psat(p)).unsqueeze(-1))
        return liq_Prsat.squeeze(-1)

    def liq_dhsatdp(self, p):
        liq_dhsatdp = torch.bmm(self.coeff_liq_dhsatdp.unsqueeze(0).repeat(len(p),1,1), self.poly8(self.norm_psat(p)).unsqueeze(-1))
        return liq_dhsatdp.squeeze(-1)

    def liq_dDsatdp(self, p):
        liq_dDsatdp = torch.bmm(self.coeff_liq_dDsatdp.unsqueeze(0).repeat(len(p),1,1), self.poly8(self.norm_psat(p)).unsqueeze(-1))
        return liq_dDsatdp.squeeze(-1)

    # Vapor region properties
    def vap_sph(self, p, h):
        vap_sph = torch.bmm(self.coeff_vap_sph.unsqueeze(0).repeat(len(p),1,1), self.poly55(self.norm_vap_p(p), self.norm_vap_h(h)).unsqueeze(-1))
        return vap_sph.squeeze(-1)

    def vap_hps(self, p, s):
        vap_hps = torch.bmm(self.coeff_vap_hps.unsqueeze(0).repeat(len(p),1,1), self.poly55(self.norm_vap_p(p), self.norm_vap_s(s)).unsqueeze(-1))
        return vap_hps.squeeze(-1)

    def vap_Dph(self, p, h):
        vap_Dph = torch.bmm(self.coeff_vap_Dph.unsqueeze(0).repeat(len(p),1,1), self.poly24(self.norm_vap_p(p), self.norm_vap_h(h)).unsqueeze(-1))
        return vap_Dph.squeeze(-1)

    def vap_muph(self, p, h):
        vap_muph = torch.bmm(self.coeff_vap_muph.unsqueeze(0).repeat(len(p),1,1), self.poly52(self.norm_vap_p(p), self.norm_vap_h(h)).unsqueeze(-1))
        return vap_muph.squeeze(-1)

    def vap_kph(self, p, h):
        vap_kph = torch.bmm(self.coeff_vap_kph.unsqueeze(0).repeat(len(p),1,1), self.poly54(self.norm_vap_p(p), self.norm_vap_h(h)).unsqueeze(-1))
        return vap_kph.squeeze(-1)

    def vap_Cph(self, p, h):
        vap_Cph = torch.bmm(self.coeff_vap_Cph.unsqueeze(0).repeat(len(p),1,1), self.poly54(self.norm_vap_p(p), self.norm_vap_h(h)).unsqueeze(-1))
        return vap_Cph.squeeze(-1)

    def vap_Tph(self, p, h):
        vap_Tph = torch.bmm(self.coeff_vap_Tph.unsqueeze(0).repeat(len(p),1,1), self.poly23(self.norm_vap_p(p), self.norm_vap_h(h)).unsqueeze(-1))
        return vap_Tph.squeeze(-1)

    def vap_CpT(self, p, T):
        vap_CpT = torch.bmm(self.coeff_vap_CpT.unsqueeze(0).repeat(len(p),1,1), self.poly55(self.norm_vap_p(p), self.norm_vap_T(T)).unsqueeze(-1))
        return vap_CpT.squeeze(-1)

    def vap_Prph(self, p, h):
        vap_Prph = torch.bmm(self.coeff_vap_Prph.unsqueeze(0).repeat(len(p),1,1), self.poly55(self.norm_vap_p(p), self.norm_vap_h(h)).unsqueeze(-1))
        return vap_Prph.squeeze(-1)

    def vap_dsdp(self, p, h):
        vap_dsdp = torch.bmm(self.coeff_vap_dsdp.unsqueeze(0).repeat(len(p),1,1), self.poly45(self.norm_vap_p(p), self.norm_vap_h(h)).unsqueeze(-1))
        return vap_dsdp.squeeze(-1)

    def vap_dsdh(self, p, h):
        vap_dsdh = torch.bmm(self.coeff_vap_dsdh.unsqueeze(0).repeat(len(p),1,1), self.poly54(self.norm_vap_p(p), self.norm_vap_h(h)).unsqueeze(-1))
        return vap_dsdh.squeeze(-1)

    def vap_dhdp(self, p, s):
        vap_dhdp = torch.bmm(self.coeff_vap_dhdp.unsqueeze(0).repeat(len(p),1,1), self.poly45(self.norm_vap_p(p), self.norm_vap_s(s)).unsqueeze(-1))
        return vap_dhdp.squeeze(-1)

    def vap_dhds(self, p, s):
        vap_dhds = torch.bmm(self.coeff_vap_dhds.unsqueeze(0).repeat(len(p),1,1), self.poly54(self.norm_vap_p(p), self.norm_vap_s(s)).unsqueeze(-1))
        return vap_dhds.squeeze(-1)

    def vap_dDdp(self, p, h):
        vap_dDdp = torch.bmm(self.coeff_vap_dDdp.unsqueeze(0).repeat(len(p),1,1), self.poly14(self.norm_vap_p(p), self.norm_vap_h(h)).unsqueeze(-1))
        return vap_dDdp.squeeze(-1)

    def vap_dDdh(self, p, h):
        vap_dDdh = torch.bmm(self.coeff_vap_dDdh.unsqueeze(0).repeat(len(p),1,1), self.poly23(self.norm_vap_p(p), self.norm_vap_h(h)).unsqueeze(-1))
        return vap_dDdh.squeeze(-1)

    # Liquid region properties
    def liq_Dph(self, p, h):
        return torch.bmm(self.coeff_liq_Dph.unsqueeze(0).repeat(len(p),1,1), self.poly25(self.norm_liq_p(p), self.norm_liq_h(h)).unsqueeze(-1)).squeeze(-1)

    def liq_muph(self, p, h):
        return torch.bmm(self.coeff_liq_muph.unsqueeze(0).repeat(len(p),1,1), self.poly24(self.norm_liq_p(p), self.norm_liq_h(h)).unsqueeze(-1)).squeeze(-1)

    def liq_kph(self, p, h):
        return torch.bmm(self.coeff_liq_kph.unsqueeze(0).repeat(len(p),1,1), self.poly24(self.norm_liq_p(p), self.norm_liq_h(h)).unsqueeze(-1)).squeeze(-1)

    def liq_Cph(self, p, h):
        return torch.bmm(self.coeff_liq_Cph.unsqueeze(0).repeat(len(p),1,1), self.poly24(self.norm_liq_p(p), self.norm_liq_h(h)).unsqueeze(-1)).squeeze(-1)

    def liq_Tph(self, p, h):
        return torch.bmm(self.coeff_liq_Tph.unsqueeze(0).repeat(len(p),1,1), self.poly24(self.norm_liq_p(p), self.norm_liq_h(h)).unsqueeze(-1)).squeeze(-1)

    def liq_CpT(self, p, T):
        return torch.bmm(self.coeff_liq_CpT.unsqueeze(0).repeat(len(p),1,1), self.poly25(self.norm_liq_p(p), self.norm_liq_T(T)).unsqueeze(-1)).squeeze(-1)

    def liq_Prph(self, p, h):
        return torch.bmm(self.coeff_liq_Prph.unsqueeze(0).repeat(len(p),1,1), self.poly25(self.norm_liq_p(p), self.norm_liq_h(h)).unsqueeze(-1)).squeeze(-1)

    def liq_dDdp(self, p, h):
        return torch.bmm(self.coeff_liq_dDdp.unsqueeze(0).repeat(len(p),1,1), self.poly15(self.norm_liq_p(p), self.norm_liq_h(h)).unsqueeze(-1)).squeeze(-1)

    def liq_dDdh(self, p, h):
        return torch.bmm(self.coeff_liq_dDdh.unsqueeze(0).repeat(len(p),1,1), self.poly24(self.norm_liq_p(p), self.norm_liq_h(h)).unsqueeze(-1)).squeeze(-1)

    
"""
import casadi as ca

class Refrigerant(object):
    def __init__(self, coefficients):
        x = ca.SX.sym("x", 1)
        y = ca.SX.sym("y", 1)
        
        # Refrigerant property fitting coefficients
        self.coefficients = coefficients
            
        self.coeff_Tsat = self.coefficients["Tsat"].reshape(-1)
        
        self.coeff_vap_hsat = self.coefficients["vap_hsat"]
        self.coeff_vap_Dsat = self.coefficients["vap_Dsat"]
        self.coeff_vap_musat = self.coefficients["vap_musat"]
        self.coeff_vap_ksat = self.coefficients["vap_ksat"]
        self.coeff_vap_Cpsat = self.coefficients["vap_Cpsat"]
        self.coeff_vap_Prsat = self.coefficients["vap_Prsat"]
        self.coeff_vap_dhsatdp = self.coefficients["vap_dhsatdp"]
        self.coeff_vap_dDsatdp = self.coefficients["vap_dDsatdp"]
        
        self.coeff_liq_hsat = self.coefficients["liq_hsat"]
        self.coeff_liq_Dsat = self.coefficients["liq_Dsat"]
        self.coeff_liq_musat = self.coefficients["liq_musat"]
        self.coeff_liq_ksat = self.coefficients["liq_ksat"]
        self.coeff_liq_Cpsat = self.coefficients["liq_Cpsat"]
        self.coeff_liq_Prsat = self.coefficients["liq_Prsat"]
        self.coeff_liq_dhsatdp = self.coefficients["liq_dhsatdp"]
        self.coeff_liq_dDsatdp = self.coefficients["liq_dDsatdp"]
        
        self.coeff_vap_sph = self.coefficients["vap_sph"]
        self.coeff_vap_hps = self.coefficients["vap_hps"]
        self.coeff_vap_Dph = self.coefficients["vap_Dph"]
        self.coeff_vap_muph = self.coefficients["vap_muph"]
        self.coeff_vap_kph = self.coefficients["vap_kph"]
        self.coeff_vap_Cph = self.coefficients["vap_Cph"]
        self.coeff_vap_Tph = self.coefficients["vap_Tph"]
        self.coeff_vap_CpT = self.coefficients["vap_CpT"]
        self.coeff_vap_Prph = self.coefficients["vap_Prph"]
        self.coeff_vap_dsdp = self.coefficients["vap_dsdp"]
        self.coeff_vap_dsdh = self.coefficients["vap_dsdh"]
        self.coeff_vap_dhdp = self.coefficients["vap_dhdp"]
        self.coeff_vap_dhds = self.coefficients["vap_dhds"]
        self.coeff_vap_dDdp = self.coefficients["vap_dDdp"]
        self.coeff_vap_dDdh = self.coefficients["vap_dDdh"]
        
        self.coeff_liq_Dph = self.coefficients["liq_Dph"]
        self.coeff_liq_muph = self.coefficients["liq_muph"]
        self.coeff_liq_kph = self.coefficients["liq_kph"]
        self.coeff_liq_Cph = self.coefficients["liq_Cph"]
        self.coeff_liq_Tph = self.coefficients["liq_Tph"]
        self.coeff_liq_CpT = self.coefficients["liq_CpT"]
        self.coeff_liq_Prph = self.coefficients["liq_Prph"]
        self.coeff_liq_dDdp = self.coefficients["liq_dDdp"]
        self.coeff_liq_dDdh = self.coefficients["liq_dDdh"]
        
        # Arbitrary vectors for fitting
        self.log_vec = ca.Function("log_vec", [x], [ca.vcat([ca.log(x), 1])])
        
        self.poly7 = ca.Function("poly7", [x], [ca.vcat([x**7, x**6, x**5, x**4, x**3, x**2, x, 1])])
        self.poly8 = ca.Function("poly8", [x], [ca.vcat([x**8, x**7, x**6, x**5, x**4, x**3, x**2, x, 1])])
        self.poly9 = ca.Function("poly9", [x], [ca.vcat([x**9, x**8, x**7, x**6, x**5, x**4, x**3, x**2, x, 1])])
        
        self.poly14 = ca.Function("poly14", [x, y], [ca.vcat([1, x, y, x*y, y**2, x*(y**2), y**3, x*(y**3), y**4])])
        self.poly15 = ca.Function("poly15", [x, y], [ca.vcat([1, x, y, x*y, y**2, x*(y**2), y**3, x*(y**3), y**4, x*(y**4), y**5])])
        self.poly23 = ca.Function("poly23", [x, y], [ca.vcat([1, x, y, x**2, x*y, y**2, (x**2)*y, x*(y**2), y**3])])
        self.poly24 = ca.Function("poly24", [x, y], [ca.vcat([1, x, y, x**2, x*y, y**2, (x**2)*y, x*(y**2), y**3, (x**2)*(y**2), x*(y**3), y**4])])
        self.poly25 = ca.Function("poly25", [x, y], [ca.vcat([1, x, y, x**2, x*y, y**2, (x**2)*y, x*(y**2), y**3, (x**2)*(y**2), x*(y**3), y**4, (x**2)*(y**3), x*(y**4), y**5])])
        self.poly45 = ca.Function("poly45", [x, y], [ca.vcat([1, x, y, x**2, x*y, y**2, x**3, (x**2)*y, x*(y**2), y**3, x**4, (x**3)*y, (x**2)*(y**2), x*(y**3), y**4, (x**4)*y, (x**3)*(y**2), (x**2)*(y**3), x*(y**4), y**5])])
        self.poly52 = ca.Function("poly52", [x, y], [ca.vcat([1, x, y, x**2, x*y, y**2, x**3, (x**2)*y, x*(y**2), x**4, (x**3)*y, (x**2)*(y**2), x**5, (x**4)*y, (x**3)*(y**2)])])
        self.poly54 = ca.Function("poly54", [x, y], [ca.vcat([1, x, y, x**2, x*y, y**2, x**3, (x**2)*y, x*(y**2), y**3, x**4, (x**3)*y, (x**2)*(y**2), x*(y**3), y**4, x**5, (x**4)*y, (x**3)*(y**2), (x**2)*(y**3), x*(y**4)])])
        self.poly55 = ca.Function("poly55", [x, y], [ca.vcat([1, x, y, x**2, x*y, y**2, x**3, (x**2)*y, x*(y**2), y**3, x**4, (x**3)*y, (x**2)*(y**2), x*(y**3), y**4, x**5, (x**4)*y, (x**3)*(y**2), (x**2)*(y**3), x*(y**4), y**5])])
        
        # Normalization functions for fittinig
        self.norm_psat = ca.Function("norm_psat", [x], [(x - 1670.0) / 955.3279])
        self.norm_Tsat = ca.Function("norm_Tsat", [x], [(x - 52.5643) / 33.3173])
        
        self.norm_vap_p = ca.Function("norm_vap_p", [x], [(x - 1300.9) / 939.1761])
        self.norm_vap_h = ca.Function("norm_vap_h", [x], [(x - 432.5771) / 32.0814])
        self.norm_vap_T = ca.Function("norm_vap_T", [x], [(x - 83.8534) / 34.6682])
        self.norm_vap_s = ca.Function("norm_vap_s", [x], [(x - 1.7544) / 0.1025])
        
        self.norm_liq_p = ca.Function("norm_liq_p", [x], [(x - 1984.0) / 880.9612])
        self.norm_liq_h = ca.Function("norm_liq_h", [x], [(x - 212.5470) / 49.7295])
        self.norm_liq_T = ca.Function("norm_liq_T", [x], [(x - 7.2744) / 36.7646])
        
        
    # Saturation properties
    def Tsat(self, p):
        a = self.coeff_Tsat[0]
        b = self.coeff_Tsat[1]
        c = self.coeff_Tsat[2]
        
        Tsat = b / (a - ca.log10(p)) - c
        return Tsat
    
    def vap_hsat(self, p):
        vap_hsat = self.coeff_vap_hsat @ self.poly8(self.norm_psat(p))
        return vap_hsat
    
    def vap_Dsat(self, p):
        vap_Dsat = self.coeff_vap_Dsat @ self.poly9(self.norm_psat(p))
        return vap_Dsat
    
    def vap_musat(self, p):
        vap_musat = self.coeff_vap_musat @ self.poly9(self.norm_psat(p))
        return vap_musat
    
    def vap_ksat(self, p):
        vap_ksat = self.coeff_vap_ksat @ self.poly9(self.norm_psat(p))
        return vap_ksat
    
    def vap_Cpsat(self, p):
        vap_Cpsat = self.coeff_vap_Cpsat @ self.poly9(self.norm_psat(p))
        return vap_Cpsat
    
    def vap_Prsat(self, p):
        vap_Prsat = self.coeff_vap_Prsat @ self.poly9(self.norm_psat(p))
        return vap_Prsat
    
    def vap_dhsatdp(self, p):
        vap_dhsatdp = self.coeff_vap_dhsatdp @ self.poly7(self.norm_psat(p))
        return vap_dhsatdp
    
    def vap_dDsatdp(self, p):
        vap_dDsatdp = self.coeff_vap_dDsatdp @ self.poly8(self.norm_psat(p))
        return vap_dDsatdp
    
    def liq_hsat(self, p):
        liq_hsat = self.coeff_liq_hsat @ self.poly9(self.norm_psat(p))
        return liq_hsat
    
    def liq_Dsat(self, p):
        liq_Dsat = self.coeff_liq_Dsat @ self.poly9(self.norm_psat(p))
        return liq_Dsat
    
    def liq_musat(self, p):
        liq_musat = self.coeff_liq_musat @ self.poly9(self.norm_psat(p))
        return liq_musat
    
    def liq_ksat(self, p):
        liq_ksat = self.coeff_liq_ksat @ self.poly8(self.norm_psat(p))
        return liq_ksat
    
    def liq_Cpsat(self, p):
        liq_Cpsat = self.coeff_liq_Cpsat @ self.poly9(self.norm_psat(p))
        return liq_Cpsat
    
    def liq_Prsat(self, p):
        liq_Prsat = self.coeff_liq_Prsat @ self.poly8(self.norm_psat(p))
        return liq_Prsat
    
    def liq_dhsatdp(self, p):
        liq_dhsatdp = self.coeff_liq_dhsatdp @ self.poly8(self.norm_psat(p))
        return liq_dhsatdp
    
    def liq_dDsatdp(self, p):
        liq_dDsatdp = self.coeff_liq_dDsatdp @ self.poly8(self.norm_psat(p))
        return liq_dDsatdp
    
    
    # Vapor region properties
    def vap_sph(self, p, h):
        vap_sph = self.coeff_vap_sph @ self.poly55(self.norm_vap_p(p), self.norm_vap_h(h))
        return vap_sph
    
    def vap_hps(self, p, s):
        vap_hps = self.coeff_vap_hps @ self.poly55(self.norm_vap_p(p), self.norm_vap_s(s))
        return vap_hps
    
    def vap_Dph(self, p, h):
        vap_Dph = self.coeff_vap_Dph @ self.poly24(self.norm_vap_p(p), self.norm_vap_h(h))
        return vap_Dph
    
    def vap_muph(self, p, h):
        vap_muph = self.coeff_vap_muph @ self.poly52(self.norm_vap_p(p), self.norm_vap_h(h))
        return vap_muph
    
    def vap_kph(self, p, h):
        vap_kph = self.coeff_vap_kph @ self.poly54(self.norm_vap_p(p), self.norm_vap_h(h))
        return vap_kph
    
    def vap_Cph(self, p, h):
        vap_Cph = self.coeff_vap_Cph @ self.poly54(self.norm_vap_p(p), self.norm_vap_h(h))
        return vap_Cph
    
    def vap_Tph(self, p, h):
        vap_Tph = self.coeff_vap_Tph @ self.poly23(self.norm_vap_p(p), self.norm_vap_h(h))
        return vap_Tph
    
    def vap_CpT(self, p, T):
        vap_CpT = self.coeff_vap_CpT @ self.poly55(self.norm_vap_p(p), self.norm_vap_T(T))
        return vap_CpT
    
    def vap_Prph(self, p, h):
        vap_Prph = self.coeff_vap_Prph @ self.poly55(self.norm_vap_p(p), self.norm_vap_h(h))
        return vap_Prph
    
    def vap_dsdp(self, p, h):
        vap_dsdp = self.coeff_vap_dsdp @ self.poly45(self.norm_vap_p(p), self.norm_vap_h(h))
        return vap_dsdp
    
    def vap_dsdh(self, p, h):
        vap_dsdh = self.coeff_vap_dsdh @ self.poly54(self.norm_vap_p(p), self.norm_vap_h(h))
        return vap_dsdh
    
    def vap_dhdp(self, p, s):
        vap_dhdp = self.coeff_vap_dhdp @ self.poly45(self.norm_vap_p(p), self.norm_vap_s(s))
        return vap_dhdp
    
    def vap_dhds(self, p, s):
        vap_dhds = self.coeff_vap_dhds @ self.poly54(self.norm_vap_p(p), self.norm_vap_s(s))
        return vap_dhds
    
    def vap_dDdp(self, p, h):
        vap_dDdp = self.coeff_vap_dDdp @ self.poly14(self.norm_vap_p(p), self.norm_vap_h(h))
        return vap_dDdp
    
    def vap_dDdh(self, p, h):
        vap_dDdh = self.coeff_vap_dDdh @ self.poly23(self.norm_vap_p(p), self.norm_vap_h(h))
        return vap_dDdh
    
    
    # Liquid region properties
    def liq_Dph(self, p, h):
        liq_Dph = self.coeff_liq_Dph @ self.poly25(self.norm_liq_p(p), self.norm_liq_h(h))
        return liq_Dph
    
    def liq_muph(self, p, h):
        liq_muph = self.coeff_liq_muph @ self.poly24(self.norm_liq_p(p), self.norm_liq_h(h))
        return liq_muph
    
    def liq_kph(self, p, h):
        liq_kph = self.coeff_liq_kph @ self.poly24(self.norm_liq_p(p), self.norm_liq_h(h))
        return liq_kph
    
    def liq_Cph(self, p, h):
        liq_Cph = self.coeff_liq_Cph @ self.poly24(self.norm_liq_p(p), self.norm_liq_h(h))
        return liq_Cph
    
    def liq_Tph(self, p, h):
        liq_Tph = self.coeff_liq_Tph @ self.poly24(self.norm_liq_p(p), self.norm_liq_h(h))
        return liq_Tph
    
    def liq_CpT(self, p, T):
        liq_CpT = self.coeff_liq_CpT @ self.poly25(self.norm_liq_p(p), self.norm_liq_T(T))
        return liq_CpT
    
    def liq_Prph(self, p, h):
        liq_Prph = self.coeff_liq_Prph @ self.poly25(self.norm_liq_p(p), self.norm_liq_h(h))
        return liq_Prph
    
    def liq_dDdp(self, p, h):
        liq_dDdp = self.coeff_liq_dDdp @ self.poly15(self.norm_liq_p(p), self.norm_liq_h(h))
        return liq_dDdp
    
    def liq_dDdh(self, p, h):
        liq_dDdh = self.coeff_liq_dDdh @ self.poly24(self.norm_liq_p(p), self.norm_liq_h(h))
        return liq_dDdh
"""