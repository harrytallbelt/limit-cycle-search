import datetime
import asyncio
from concurrent.futures import ProcessPoolExecutor # TODO: Why exacly not thread pool?
import numpy as np
from scipy.integrate import odeint
from tsanalysis import EndBehaviour, determine_end_behaviour


THREADS_COUNT = 2
COMMON_LOG_FILE_NAME = 'general.log'
CYCLES_LOG_FILE_NAME = 'cycles.log'
STRANGE_ATTRACTORS_LOG_FILE_NAME = 'strange-attrators.log'


# Legend for N_dot:
# -----------------------------------------------------------------------
# | a[0]      | a[1]    | a[2] | a[3] | a[4] | a[5]  | a[6]      | a[7] |
# |-----------|---------|------|------|------|-------|-----------|------|
# | epsilon_1 | gamma_1 |  d   |  L   |  K   | alpha | epsilon_3 |  k   |
# -----------------------------------------------------------------------

def N_dot(N, t, a):
    N_dot_1 = (a[0] - a[1] * N[0]) * N[0] - a[2] * (N[0] - N[1])
    N_dot_2 = a[0] * N[1] * (N[1] - a[3]) * (a[4] - N[1]) / a[4] \
            - a[5] * N[1] * N[2] + a[2] * (N[0] - N[1])
    N_dot_3 = -a[6] * N[2] + a[7] * a[5] * N[1] * N[2]
    return np.array([N_dot_1, N_dot_2, N_dot_3])



def analyse_random_parameters(param_max, param_min, initial_states, time_interval):
    param_range = (param_max - param_min)
    params = param_min + np.random.rand(param_count) * param_range
    fixed_params_N_dot = lambda N, t: N_dot(N, t, params)
    end_behaviours = []
    
    for N_init in initial_states:
        time_series = odeint(fixed_params_N_dot, N_init, time_interval)
        end_behaviour = determine_end_behaviour(time_series)
        end_behaviours.append(end_behaviour)
    
    return params, end_behaviours


async def start_search_chain(param_max, param_min, initial_states, time_interval, loop, pool):
    params, end_behaviours = await loop.run_in_executor(
        pool, analyse_random_parameters,
        param_max, param_min, initial_states, time_interval
    )
    # TODO: Err.. but do we have to synchronise these lines?
    # This would be the case if the event loop queue isn't thread safe.
    # If so, what is the point of using the event loop anyway?
    loop.put_nowait(lambda: on_new_results(params, initial_states, end_behaviours))
    loop.put_nowait(start_search_chain)


# Keeping track of total number of systems in console.
_total, _cycles, _strange = 0, 0, 0

def on_new_results(params, initial_states, end_behaviours):
    assert(len(initial_states) == len(end_behaviours))
    
    # TODO: Rewrite it in three loops so we only
    # have three file opennings instead of n to 2n.
    for initial_state, end_behaviour in zip(initial_states, end_behaviours):
        time_string = str(datetime.datetime.now()).split('.')[0]
        log_entity = {
            'time': time_string,
            'end_behaviour': end_behaviour,
            'parameters': parameters,
            'initial_state': initial_state
        }
        with open(COMMON_LOG_FILE_NAME, 'w+') as common_log:
            common_log.write(str(log_entity))
        _total += 1

        # Don't need that for behaviour-specific logs.
        del log_entity['end_behaviour']
        
        if end_behaviour == EndBehaviour.LIMIT_CYCLE:
            with open(CYCLES_LOG_FILE_NAME, 'w+') as cycle_log:
                cycle_log.write(str(log_entity))
            _cycles += 1
        
        elif end_behaviour == EndBehaviour.STRANGE_ATTRACTOR:
            with open(STRANGE_ATTRACTORS_LOG_FILE_NAME, 'w+') as strange_log:
                strange_log.write(str(log_entity))
            _strange += 1
    
    # This is intentionaly outside the loop.
    # Console IO isn't fast and we don't need update that frequent.
    print(f'\rSystems in total: {_total}; cycles: {_cycles}, strange attractors: {_strange}.')


# TODO: MAN, just use threads and a mutex
# on the only function dealing with IO.
# I BEG YOU.
def search():
    loop = asyncio.get_event_loop()
    pool = ProcessPoolExecutor(THREADS_COUNT)
    
    # TODO: It appears L doesn't have to be positive.
    params_min, params_max = 0.1, 100
    params_count = 8
    
    # You don't want start at origin as it is always
    # an equilibrium point (small populations die out).
    interval = np.arange(0.1, 1 + 0.01, 0.3)
    initial_states = [[x, y, z]
        for x in interval for y in interval for z in interval]
    time_interval = np.arange(0, 100, 0.1)

    start_search = lambda: start_search_chain(
        params_max, params_min,
        initial_states, time_interval,
        loop, pool
    )
    searches = [start_search() for _ in range(THREADS_COUNT)]
    loop.run_until_complete(searches)



if __name__ == '__main__':
    search()
