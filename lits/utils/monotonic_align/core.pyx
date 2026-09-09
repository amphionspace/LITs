import numpy as np

cimport cython
cimport numpy as np

from cython.parallel import prange
from libc.stdlib cimport free, malloc


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void maximum_path_each(int[:,::1] path, float[:,::1] value, int t_x, int t_y, float max_neg_val) nogil:
  cdef int x
  cdef int y
  cdef float v_prev
  cdef float v_cur
  cdef float tmp
  cdef int index = t_x - 1

  for y in range(t_y):
    for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
      if x == y:
        v_cur = max_neg_val
      else:
        v_cur = value[x, y-1]
      if x == 0:
        if y == 0:
          v_prev = 0.
        else:
          v_prev = max_neg_val
      else:
        v_prev = value[x-1, y-1]
      value[x, y] = max(v_cur, v_prev) + value[x, y]

  for y in range(t_y - 1, -1, -1):
    path[index, y] = 1
    if index != 0 and (index == y or value[index, y-1] < value[index-1, y-1]):
      index = index - 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void maximum_path_c(int[:,:,::1] paths, float[:,:,::1] values, int[::1] t_xs, int[::1] t_ys, float max_neg_val=-1e9) nogil:
  cdef int b = values.shape[0]

  cdef int i
  for i in prange(b, nogil=True):
    maximum_path_each(paths[i], values[i], t_xs[i], t_ys[i], max_neg_val)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int _ceil_limit(int ceiling, int t_y) nogil:
  if ceiling <= 0:
    return t_y
  return ceiling


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _deque_push(
    int* deque_yp, float* deque_score, int* deque_head, int* deque_tail,
    int yp, float score,
) nogil:
  cdef int tail = deque_tail[0]
  while tail > deque_head[0]:
    if deque_score[tail - 1] <= score:
      tail -= 1
    else:
      break
  deque_yp[tail] = yp
  deque_score[tail] = score
  deque_tail[0] = tail + 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _deque_expire(int* deque_yp, int* deque_head, int* deque_tail, int min_yp) nogil:
  while deque_head[0] < deque_tail[0] and deque_yp[deque_head[0]] < min_yp:
    deque_head[0] += 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _best_from_deque(
    int* deque_yp, float* deque_score, int deque_head, int deque_tail,
    float valid_threshold, float* best, int* arg,
) nogil:
  if deque_head < deque_tail and deque_score[deque_head] > valid_threshold:
    best[0] = deque_score[deque_head]
    arg[0] = deque_yp[deque_head]
  else:
    best[0] = -1e9
    arg[0] = -1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void maximum_path_constrained_each(int[:,::1] path, float[:,::1] value, int[::1] floors, int[::1] ceilings,
                                        int t_x, int t_y, float max_neg_val,
                                        float* score_constrained, float* score_free) nogil:
  """Monotonic alignment with per-token minimum and maximum durations.

  DP over E[x, y] = best score of any monotonic path in which token x starts
  exactly at frame y. Token x-1 must span between floor and ceiling frames:

      floor[x-1] <= y - y' <= ceiling[x-1]

  where y' is the start frame of token x-1. ceiling[i] <= 0 means unbounded.
  """
  cdef int x
  cdef int y
  cdef int f
  cdef int c
  cdef int cand
  cdef int arg_c
  cdef float best_c
  cdef float best_f
  cdef float g
  cdef float s_prev
  cdef float valid_threshold = max_neg_val * 0.5
  cdef float *e_c_prev = <float*> malloc(t_y * sizeof(float))
  cdef float *e_c_cur = <float*> malloc(t_y * sizeof(float))
  cdef float *e_f_prev = <float*> malloc(t_y * sizeof(float))
  cdef float *e_f_cur = <float*> malloc(t_y * sizeof(float))
  cdef int *amax = <int*> malloc(t_x * t_y * sizeof(int))
  cdef int *starts = <int*> malloc(t_x * sizeof(int))
  cdef float *tmp
  cdef int *deque_yp_c = <int*> malloc(t_y * sizeof(int))
  cdef float *deque_score_c = <float*> malloc(t_y * sizeof(float))
  cdef int *deque_yp_f = <int*> malloc(t_y * sizeof(int))
  cdef float *deque_score_f = <float*> malloc(t_y * sizeof(float))
  cdef int c_head
  cdef int c_tail
  cdef int f_head
  cdef int f_tail
  cdef int dummy_arg
  cdef int min_yp

  # Per-row prefix sums: value[x, y] becomes S[x][y] = sum(value[x, 0..y]).
  for x in range(t_x):
    for y in range(1, t_y):
      value[x, y] += value[x, y - 1]

  # Token 0 can only start at frame 0.
  for y in range(t_y):
    e_c_prev[y] = max_neg_val
    e_f_prev[y] = max_neg_val
  e_c_prev[0] = 0.0
  e_f_prev[0] = 0.0

  for x in range(1, t_x):
    f = floors[x - 1]
    if f < 1:
      f = 1
    c = _ceil_limit(ceilings[x - 1], t_y)
    c_head = 0
    c_tail = 0
    f_head = 0
    f_tail = 0
    for y in range(t_y):
      # Constrained window: candidate y' = y - f must satisfy y - c <= y' <= y - f.
      cand = y - f
      if cand >= 0 and e_c_prev[cand] > valid_threshold:
        s_prev = value[x - 1, cand - 1] if cand > 0 else 0.0
        g = e_c_prev[cand] - s_prev
        _deque_push(deque_yp_c, deque_score_c, &c_head, &c_tail, cand, g)
      min_yp = y - c
      _deque_expire(deque_yp_c, &c_head, &c_tail, min_yp)
      _best_from_deque(deque_yp_c, deque_score_c, c_head, c_tail, valid_threshold, &best_c, &arg_c)

      # Free (min-1) window with the same ceiling on token x-1.
      cand = y - 1
      if cand >= 0 and e_f_prev[cand] > valid_threshold:
        s_prev = value[x - 1, cand - 1] if cand > 0 else 0.0
        g = e_f_prev[cand] - s_prev
        _deque_push(deque_yp_f, deque_score_f, &f_head, &f_tail, cand, g)
      min_yp = y - c
      _deque_expire(deque_yp_f, &f_head, &f_tail, min_yp)
      _best_from_deque(deque_yp_f, deque_score_f, f_head, f_tail, valid_threshold, &best_f, &dummy_arg)

      amax[x * t_y + y] = arg_c
      if arg_c >= 0:
        e_c_cur[y] = best_c + value[x - 1, y - 1]
      else:
        e_c_cur[y] = max_neg_val
      if best_f > valid_threshold:
        e_f_cur[y] = best_f + value[x - 1, y - 1]
      else:
        e_f_cur[y] = max_neg_val
    tmp = e_c_prev; e_c_prev = e_c_cur; e_c_cur = tmp
    tmp = e_f_prev; e_f_prev = e_f_cur; e_f_cur = tmp

  # Final (virtual) step: the last token must cover frames start..t_y-1.
  f = floors[t_x - 1]
  if f < 1:
    f = 1
  c = _ceil_limit(ceilings[t_x - 1], t_y)
  best_c = max_neg_val
  best_f = max_neg_val
  arg_c = -1
  for cand in range(max(0, t_y - c), t_y - f + 1):
    if e_c_prev[cand] > valid_threshold:
      s_prev = value[t_x - 1, cand - 1] if cand > 0 else 0.0
      g = e_c_prev[cand] - s_prev
      if g > best_c:
        best_c = g
        arg_c = cand
  for cand in range(max(0, t_y - c), t_y):
    if e_f_prev[cand] > valid_threshold:
      s_prev = value[t_x - 1, cand - 1] if cand > 0 else 0.0
      g = e_f_prev[cand] - s_prev
      if g > best_f:
        best_f = g
  score_constrained[0] = best_c + value[t_x - 1, t_y - 1]
  score_free[0] = best_f + value[t_x - 1, t_y - 1]

  if arg_c >= 0:
    starts[t_x - 1] = arg_c
    for x in range(t_x - 1, 0, -1):
      starts[x - 1] = amax[x * t_y + starts[x]]
    for x in range(t_x):
      cand = starts[x + 1] if x + 1 < t_x else t_y
      for y in range(starts[x], cand):
        path[x, y] = 1

  free(e_c_prev)
  free(e_c_cur)
  free(e_f_prev)
  free(e_f_cur)
  free(amax)
  free(starts)
  free(deque_yp_c)
  free(deque_score_c)
  free(deque_yp_f)
  free(deque_score_f)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void maximum_path_constrained_c(int[:,:,::1] paths, float[:,:,::1] values, int[:,::1] floors, int[:,::1] ceilings,
                                      int[::1] t_xs, int[::1] t_ys,
                                      float[::1] scores_constrained, float[::1] scores_free,
                                      float max_neg_val=-1e9) nogil:
  cdef int b = values.shape[0]

  cdef int i
  for i in prange(b, nogil=True):
    if t_xs[i] <= 0 or t_ys[i] < t_xs[i]:
      # Degenerate sample: not enough frames for one per token. Fall back to
      # the classic kernel so behaviour matches the unconstrained code path.
      maximum_path_each(paths[i], values[i], t_xs[i], t_ys[i], max_neg_val)
      scores_constrained[i] = 0.0
      scores_free[i] = 0.0
    else:
      maximum_path_constrained_each(paths[i], values[i], floors[i], ceilings[i], t_xs[i], t_ys[i],
                                    max_neg_val, &scores_constrained[i], &scores_free[i])
