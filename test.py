from esinet.minimum_norm import mne_eloreta
from esinet.simulation import Simulation
from esinet.forward import create_forward_model, get_info
from esinet.net import Net
from esinet.util import wrap_mne_inverse, unpack_fwd

info = get_info()
info['sfreq'] = 100
fwd = create_forward_model(info=info)
pos = unpack_fwd(fwd)[2]

settings = dict(duration_of_trial=0.2)
sim = Simulation(fwd, info, settings=settings)
sim.simulate(n_samples=10)
print(sim.simulation_info)
