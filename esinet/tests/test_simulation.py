import pytest
from .. import simulation
from .. import forward

fwd = forward.create_forward_model(sampling='ico3')
info = forward.get_info()

def test_create_fwd_model():
    sampling = 'ico1'
    fwd = forward.create_forward_model(sampling=sampling)

def test_create_info():
    info = forward.get_info()


@pytest.mark.parametrize("number_of_sources", [2,])
@pytest.mark.parametrize("extents", [(1,20),])
@pytest.mark.parametrize("amplitudes", [1,])
@pytest.mark.parametrize("shapes", ['mixed', 'gaussian', 'flat'])
@pytest.mark.parametrize("duration_of_trial", [0, 0.1, (0, 0.1)])
@pytest.mark.parametrize("sample_frequency", [100,])
@pytest.mark.parametrize("target_snr", [5, ])
@pytest.mark.parametrize("beta", [1, ])
@pytest.mark.parametrize("method", ['standard', 'noise', 'mixed'])
@pytest.mark.parametrize("source_spread", ['region_growing', 'spherical', 'mixed'])
@pytest.mark.parametrize("source_number_weighting", [True, False])
@pytest.mark.parametrize("parallel", [False])
def test_simulation( number_of_sources, extents, amplitudes,
        shapes, duration_of_trial, sample_frequency, target_snr, beta,
        method, source_spread, source_number_weighting, parallel):

    settings = {
            'number_of_sources': number_of_sources,
            'extents': extents,
            'amplitudes': amplitudes,
            'shapes': shapes,
            'duration_of_trial': duration_of_trial,
            'sample_frequency': sample_frequency,
            'target_snr': target_snr,
            'beta': beta,
            'method': method,
            'source_spread': source_spread,
            'source_number_weighting': source_number_weighting
        }

    sim = simulation.Simulation(fwd, info, settings=settings, parallel=parallel)
    sim.simulate(n_samples=2)


def test_simulation_add():
    sim_a = simulation.Simulation(fwd, info).simulate(n_samples=2)
    sim_b = simulation.Simulation(fwd, info).simulate(n_samples=2)
    sim_c = sim_a + sim_b
    

def create_forward_model_test(pth_fwd='temp/ico2/', sampling='ico2'):
    # Create a forward model
    forward.create_forward_model(pth_fwd, sampling=sampling)
    info = forward.get_info(sfreq=100)
    fwd = forward.create_forward_model(info=info)
    fwd_free = forward.create_forward_model(info=info, fixed_ori=False)
        
 
    
