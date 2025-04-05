function x_calc = x_next_calculation(x_now, u_now, p_now, Ref, Cool_evap, setting_evap)
   
sys = @(t, x) sys_evap_matlab(x, u_now, p_now, Ref, Cool_evap, setting_evap);
[~, res_x] = ode15s(sys, [0, setting_evap.config.time_interval], x_now);

x_calc = res_x(end, :)';

end