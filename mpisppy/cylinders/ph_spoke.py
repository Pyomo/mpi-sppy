from mpisppy.cylinders.spoke import Spoke, Field
import pyomo.environ as pyo

class PHSpoke(Spoke):
    send_fields = (*Spoke.send_fields, Field.XFEAS)
    receive_fields = (*Spoke.receive_fields, )

    def update_rho(self):
        rho_factor = self.opt.options.get("rho_factor", 1.0)
        if rho_factor == 1.0:
            return
        for s in self.opt.local_scenarios.values():
            for rho in s._mpisppy_model.rho.values():
                rho._value = rho_factor * rho._value

    def main(self):
        # setup, PH Iter0
        smoothed = self.options.get('smoothed', 0)
        attach_prox = True
        self.opt.PH_Prep(attach_prox=attach_prox, attach_smooth = smoothed)
        trivial_bound = self.opt.Iter0()

        # update the rho
        self.update_rho()

        # rest of PH
        self.opt.iterk_loop()

        return self.opt.conv, None, trivial_bound
    
    
    def send_xfeas(self):
        xfeas_buf = self.send_buffers[Field.XFEAS]
        ci = 0
        for (sname, s) in self.opt.local_scenarios.items():
            for xvar in s._mpisppy_data.nonant_indices.values():
                xfeas_buf[ci] = xvar._value
                ci += 1
            self.opt.disable_W_and_prox()
            objfct = self.opt.saved_objectives[sname]
            xfeas_buf[ci] = pyo.value(objfct)
            self.opt.reenable_W_and_prox()
            ci += 1
        self.put_send_buffer(xfeas_buf, Field.XFEAS)


    def sync(self):
       
        self.send_xfeas()
        # Update the nonant bounds, if possible
        self.receive_nonant_bounds()

    def finalize(self):
        if self.opt.best_bound_obj_val is None:
            return

        # Tell the hub about the most recent bound
        #self.send_bound(self.opt.best_bound_obj_val)
        self.final_bound = self.opt.best_bound_obj_val

        return self.final_bound
