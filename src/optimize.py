# Cost fcn

"""
Robust MPC
- 방출부의 온도를 특정 온도로 유지하는 것 (Quadratic 목적 함수 사용을 통한 reference value tracking problem)
- (Evaporator의 경우) 방출부 냉매의 상을 기상으로 만들어주기 위한 최소한의 외부 열량을 공급하는 것 (Economic objective optimization problem)


Q = cost fcn = eps_tp * m_ref_in * (T_util_in - T_in_ref)
max(x) min(u) Q through gradient descent of PINN model

constraints
- abs(T_ref_out - T_ref_out_true) < threshold (barrier method)"
"""