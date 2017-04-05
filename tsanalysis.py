from enum import Enum
from numpy.linalg import norm as np_norm


class EndBehaviour(Enum):
    LIMIT_POINT = 0
    LIMIT_CYCLE = 1
    INFINITY = 2
    STRANGE_ATTRACTOR = 3


DEFAULD_CONFIDENCE_COUNT = 20
DEFAULT_CLOSE_DISTANCE = 0.001


def determine_end_behaviour(time_series,
                            norm=np_norm,
                            n=DEFAULD_CONFIDENCE_COUNT,
                            eps=DEFAULT_CLOSE_DISTANCE):
    """
    Determines the end behaviour of the given time series.
    Args:
        time_series: An array-like of vectors - array-likes
            of the same size.
        
        norm: The norm, used to calculate vector distances.
            Defaults to np.linalg.norm.
        
        n: The function considers `n` last vectors of time_series.
            Defaults to DEFAULD_CONFIDENCE_COUNT.

        eps: The close enough distance between two vectors.
            Defaults to DEFAULT_CLOSE_DISTANCE.
    Returns:
        One of the EndBehaviour enum values.
    
    Raises:
        ValueError: if the time series is too short for the choosen n.
    """
    if goes_to_limit_point(time_series, norm, n, eps):
        return EndBehaviour.LIMIT_POINT

    if goes_to_infinity(time_series, norm, n, eps):
        return EndBehaviour.INFINITY
    
    if goes_to_limit_cycle(time_series, norm, n, eps):
        return EndBehaviour.LIMIT_CYCLE
    # If it is none of above,
    # it must be a strange attrator.
    return EndBehaviour.STRANGE_ATTRACTOR



def goes_to_limit_point(time_series,
                        norm=np_norm,
                        n=DEFAULD_CONFIDENCE_COUNT,
                        eps=DEFAULT_CLOSE_DISTANCE):
    # Take the last point of the time series ...
    success_count = 0
    reversed_time_series = reversed(time_series)
    limit_point = next(reversed_time_series)
    # ... and check that n last points
    # are close enought to it.
    for v in reversed_time_series:
        if norm(limit_point - v) > eps:
            return False
        success_count += 1
        next_vector = v
        if success_count == n:
            return True
    # The exception is raised if n > len(time_series).
    raise ValueError('The time series is too short for '
                   + f'the choosen confidence count n={n}.')


# TODO: What about the case when distances are decreasing,
# but the function in still unbounded? Like log(x) or the harmonic series.
# The solution might be to consider not the distances between the dots,
# but the distances between the final dot and all the others backwards.
# If those distances increase (supposing the time series is long enough),
# then it is definetely tends to infinity.
# We didn't use eps in the last take on it, but do we need it now?
def goes_to_infinity(time_series,
                     norm=np_norm,
                     n=DEFAULD_CONFIDENCE_COUNT):
    success_count = 0
    reversed_time_series = reversed(time_series)
    # Find the distance between the last two points ...
    next_vector = next(reversed_time_series)
    current_vector = next(reversed_time_series)
    next_distance = norm(next_vector - current_vector)
    next_vector = current_vector
    # ... and make sure it only decreases
    # (as time decreases) for n last points.
    for current_vector in reversed_time_series:
        current_distance = norm(next_vector - current_vector)
        if current_distance >= next_distance:
            return False
        success_count += 1
        next_vector = current_vector
        next_distance = current_distance
        if success_count == n:
            return True
    # The exception is raised if n > len(time_series).
    raise ValueError('The time series is too short for '
                    + f'the choosen confidence count n={n}.')



def goes_to_limit_cycle(time_series,
                        norm=np_norm,
                        n=DEFAULD_CONFIDENCE_COUNT,
                        eps=DEFAULT_CLOSE_DISTANCE):
    success_count = 0
    # For each point p1 (from the end)
    for i in range(len(time_series), 0, -1):
        # For each point p2 before it
        for j in range(i - 1, 0, -1):
            # If p1 and p2 close enough,
            # they might be a part of a cycle.
            if norm(time_series[i] - time_series[j]) < eps:
                # Suppose they are a part of cycle with
                # a period of their distance in time series.
                cycle_flag, period = True, i - j
                # Walk backwards with this period and check
                # if all the points will be close enough.
                curr = time_series[j]
                for k in range(j - period, 0, -period):
                    next, curr = curr, time_series[k]
                    if norm(curr - next) > eps:
                        cycle_flag = False
                        break
                # If they are, consider it a good
                # evidence for a limit cycle.
                if cycle_flag:
                    success_count += 1
    # If we return False here, it might be
    # that we just didn't have enough information.
    return success_count >= n
