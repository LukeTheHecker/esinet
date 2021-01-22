import shutil
from ESINet.tests.unit_tests import *



def run_unit_tests():
    pth_fwd='temp/ico2/'
    sampling = 'ico2'

    
    print(f'\nTesting forward model...')
    result = create_forward_model_test(pth_fwd=pth_fwd, sampling=sampling)
    print('\tPassed') if result else print('\tFailed')
    
    print(f'\nTesting Simulations...')
    result = run_simulations_test(pth_fwd=pth_fwd)
    print('\tPassed') if result else print('\tFailed')
    
    # shutil.rmtree(pth_fwd)