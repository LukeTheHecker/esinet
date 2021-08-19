import pytest
from .. import simulation
from .. import forward
from .. import Net

def test_net_single_instance():
    
    # Crate forward model
    info = forward.get_info()
    sampling = 'ico3'
    fwd = forward.create_forward_model(sampling=sampling)
    
    # Simulate some little data
    settings = dict(duration_of_trial=0)
    sim = simulation.Simulation(fwd, info, settings=settings)
    sim.simulate(n_samples=10)
    # Create and train net
    ann = Net(fwd, n_neurons=1)
    ann.fit(sim)
    assert not ann.temporal, 'Instance of Net() must not be temporal here!'
    # Predict
    y = ann.predict(sim)

def test_net_temporal():
    
    # Crate forward model
    info = forward.get_info()
    sampling = 'ico3'
    fwd = forward.create_forward_model(sampling=sampling)
    
    # Simulate some little data
    settings = dict(duration_of_trial=0.2)
    sim = simulation.Simulation(fwd, info, settings=settings)
    sim.simulate(n_samples=10)
    # Create and train net
    ann = Net(fwd, n_neurons=1)
    ann.fit(sim)
    assert ann.temporal, 'Instance of Net() must be temporal here!'
    
    # Predict
    y = ann.predict(sim)
