import shutil
from ESINet.tests.unit_tests import *



def run_unit_tests():
    pth_fwd='temp/ico2/'
    sampling = 'ico2'

    results = dict()

    print(f'\nTesting forward model...')
    results['fwd'] = create_forward_model_test(pth_fwd=pth_fwd, sampling=sampling)
    print('\tPassed') if results['fwd'] else print('\tFailed')
    
    print(f'\nTesting Simulations...')
    results['sim'] = run_simulations_test(pth_fwd=pth_fwd)
    print('\tPassed') if results['sim'] else print('\tFailed')
    
    print(results)

    shutil.rmtree(pth_fwd)