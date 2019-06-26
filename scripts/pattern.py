from scripts import models, utils
import numpy as np
from enum import Enum
import re
from copy import copy
import math
from iterextras import unzip

CONTEXT = {}
SCORE = 0


def no_score(it):
    return map(lambda t: (t, 0), it)


class Node:
    def _eval(self, *args, **kwargs):
        global SCORE
        if self in CONTEXT:
            yield CONTEXT[self]
        else:
            for result, score in self._eval_inner(*args, **kwargs):
                CONTEXT[self] = result
                SCORE += score
                yield result
                del CONTEXT[self]
                SCORE -= score

    def _eval_inner(self, *args, **kwargs):
        raise NotImplemented

    def eval(self, video=None, *args, **kwargs):
        if video is not None:
            videos = [video]
        else:
            videos = models.Video.objects.all()
        contexts, scores = unzip([(copy(CONTEXT), SCORE) for video in videos
                                  for _ in self._eval(video, *args, **kwargs)])
        sorted_idx = np.argsort(scores)
        return [[contexts[i] for i in sorted_idx],
                [scores[i] for i in sorted_idx]]


class Trajectory(Node):
    def __init__(self, obj):
        self.obj = obj

    def filter(self, fn):
        return TrajectoryFilter(self, fn)

    def window(self, **kwargs):
        return TrajectoryWindow(self, **kwargs)

    def start(self):
        return TrajectoryPoint(self, 0)

    def end(self):
        return TrajectoryPoint(self, -1)

    def _eval_inner(self, video):
        for obj in self.obj._eval(video):
            yield (obj.trajectory(), 0)


class TrajectoryFilter(Trajectory, Node):
    def __init__(self, traj, fn):
        self.traj = traj
        self.fn = fn

    def _eval_inner(self, video):
        return no_score(filter(self.fn, self.traj._eval(video)))


class TrajectoryPoint(Node):
    def __init__(self, traj, i):
        self.traj = traj
        self.i = i

    def time(self):
        return TimePoint(self)

    def _eval_inner(self, video):
        for t in self.traj._eval(video):
            yield (t.pos[self.i], 0)


class SpatialPattern(Node):
    def __init__(self, time, pattern):
        self.time = time
        self.pattern = pattern

    def _eval_inner(self, video):
        for time in self.time._eval(video):
            yield from no_score(self.pattern._eval(video, time))


class TimePoint(Node):
    def __init__(self, traj_point):
        self.traj_point = traj_point

    def match(self, pattern):
        return SpatialPattern(self, pattern)

    def _eval_inner(self, video):
        for p in self.traj_point._eval(video):
            yield (p.time(), 0)


class Object(Node):
    def _eval_inner(self, video):
        yield from no_score(models.Vehicle.objects.filter(video=video))
        yield from no_score(models.Pedestrian.objects.filter(video=video))


class Vehicle(Node):
    def trajectory(self):
        return Trajectory(self)

    def _eval_inner(self, video):
        yield from no_score(models.Vehicle.objects.filter(video=video))


class Pedestrian(Node):
    def trajectory(self):
        return Trajectory(self)

    def _eval_inner(self, video):
        yield from no_score(models.Pedestrian.objects.filter(video=video))


class TrajectoryWindow(Trajectory):
    def __init__(self, traj, minlen=None, maxlen=None):
        self._traj = traj
        self._minlen = minlen
        self._maxlen = maxlen

    def _eval_inner(self, video):
        for traj in self._traj._eval(video):
            N = len(traj.pos)
            minlen = self._minlen if self._minlen is not None else 1
            maxlen = self._maxlen if self._maxlen is not None else N
            for i in range(0, N - maxlen):
                for j in range(i + minlen, min(i + maxlen, N)):
                    yield (traj.index(lambda t: t[i:j]), 0)


def slowing_down(traj, eps=0.01):
    points = traj.for_numpy()
    vels = points[::3][1:] - points[::3][:-1]
    vel_mag = np.linalg.norm(vels, axis=1)
    return vel_mag[0] > eps * 3.0 and vel_mag[-1] < eps


def is_speeding(traj):
    points = traj.for_numpy()
    vels = points[::3][1:] - points[::3][:-1]
    vel_mag = np.linalg.norm(vels, axis=1)
    batches = list(utils.make_batch(vel_mag, window_size))


class TurningLeft(Node):
    def _eval_inner(self, traj):
        turn = traj.pos[0].psi - traj.pos[-1].psi
        yield (traj, abs(turn - math.pi / 2))


class SpatialRelation(Node):
    def __init__(self, obj1, obj2):
        self.obj1 = obj1
        self.obj2 = obj2

    def _eval_inner(self, video, time):
        for obj1 in self.obj1._eval(video):
            obj1_pos = obj1.trajectory().frame(time)
            if obj1_pos is None:
                continue

            for obj2 in self.obj2._eval(video):
                obj2_pos = obj2.trajectory().frame(time)
                if obj2_pos is None:
                    continue

                if type(obj1) == type(obj2) and obj1.id == obj2.id:
                    continue

                yield (None, self._relation(obj1_pos, obj2_pos))

    def _relation(self, obj1, obj2):
        raise NotImplemented


class Close(SpatialRelation):
    def _relation(self, obj1, obj2):
        dist = np.linalg.norm(obj1.for_numpy() - obj2.for_numpy())
        return dist
