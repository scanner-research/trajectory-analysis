from scripts import models, utils
import numpy as np
from enum import Enum
import re
from copy import copy
import math
from iterextras import unzip
from tqdm import tqdm

CONTEXT = {}
SCORE = 0


def no_score(it):
    return map(lambda t: (t, 0), it)


class Node:
    def __init__(self):
        self.children = []

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

    def eval(self, video=None, progress=True, *args, **kwargs):
        if video is not None:
            if isinstance(video, list):
                videos = video
            else:
                videos = [video]
        else:
            videos = list(models.Video.objects.all())

        results = []
        for video in tqdm(videos):
            for _ in self._eval(video, *args, **kwargs):
                results.append((copy(CONTEXT), SCORE))

        contexts, scores = unzip(results)
        sorted_idx = np.argsort(scores)
        return [[contexts[i] for i in sorted_idx],
                [scores[i] for i in sorted_idx]]


class Trajectory(Node):
    def __init__(self, obj):
        self.obj = obj
        self.children = [obj]

    def where(self, fn):
        return TrajectoryWhere(self, fn)

    def weight(self, fn, **kwargs):
        return TrajectoryWeight(self, fn, **kwargs)

    def window(self, **kwargs):
        return TrajectoryWindow(self, **kwargs)

    def join(self, obj):
        return TrajectoryJoin(self, obj)

    def start(self):
        return TrajectoryPoint(self, 0)

    def end(self):
        return TrajectoryPoint(self, -1)

    def _eval_inner(self, video):
        for obj in self.obj._eval(video):
            yield (obj.trajectory(), 0)


class TrajectoryJoin(Trajectory, Node):
    def __init__(self, obj1, obj2):
        self.obj1 = obj1
        self.obj2 = obj2
        self.children = [obj1, obj2]

    def _eval_inner(self, video):
        for obj1 in self.obj1._eval(video):
            for obj2 in self.obj2._eval(video):
                yield ((obj1, obj2), 0)


class TrajectoryWhere(Trajectory, Node):
    def __init__(self, traj, fn):
        self.traj = traj
        self.fn = fn
        self.children = [traj]

    def _eval_inner(self, video):
        return no_score(filter(self.fn, self.traj._eval(video)))


class TrajectoryWeight(Trajectory, Node):
    def __init__(self, traj, fn, max=None):
        self.traj = traj
        self.fn = fn
        self.max = max
        self.children = [traj]

    def _eval_inner(self, video):
        for t in self.traj._eval(video):
            score = self.fn(t)
            if self.max is not None and score > self.max:
                continue
            yield (t, score)


class TrajectoryPoint(Node):
    def __init__(self, traj, i):
        self.traj = traj
        self.i = i
        self.children = [traj]

    def time(self):
        return TimePoint(self)

    def _eval_inner(self, video):
        for t in self.traj._eval(video):
            yield (t.pos[self.i], 0)


class SpatialPattern(Node):
    def __init__(self, time, pattern):
        self.time = time
        self.pattern = pattern
        self.children = [time, pattern]

    def _eval_inner(self, video):
        for time in self.time._eval(video):
            yield from no_score(self.pattern._eval(video, time))


class TimePoint(Node):
    def __init__(self, traj_point):
        self.traj_point = traj_point
        self.children = [traj_point]

    def match(self, pattern):
        return SpatialPattern(self, pattern)

    def _eval_inner(self, video):
        for p in self.traj_point._eval(video):
            yield (p.time(), 0)


class Object(Node):
    def __init__(self):
        self.children = []
        self.cache = {}

    def _eval_inner(self, video):
        if not video.id in self.cache:
            self.cache[video.id] = (list(
                models.Pedestrian.objects.filter(video=video)),
                                    list(
                                        models.Vehicle.objects.filter(
                                            video=video)))
        ped, veh = self.cache[video.id]
        yield from no_score(ped)
        yield from no_score(veh)


class Vehicle(Node):
    def __init__(self):
        self.children = []
        self.cache = {}

    def trajectory(self):
        return Trajectory(self)

    def _eval_inner(self, video):
        if not video.id in self.cache:
            self.cache[video.id] = list(
                models.Vehicle.objects.filter(video=video))
        yield from no_score(self.cache[video.id])


class Pedestrian(Node):
    def __init__(self):
        self.children = []
        self.cache = {}

    def trajectory(self):
        return Trajectory(self)

    def _eval_inner(self, video):
        if not video.id in self.cache:
            self.cache[video.id] = list(
                models.Pedestrian.objects.filter(video=video))
        yield from no_score(self.cache[video.id])


class TrajectoryWindow(Trajectory):
    def __init__(self, traj, minlen=None, maxlen=None, stride=None):
        self._traj = traj
        self._minlen = minlen
        self._maxlen = maxlen
        self._stride = stride
        self.children = [traj]

    def _eval_inner(self, video):
        for traj in self._traj._eval(video):
            N = len(traj.pos)
            minlen = self._minlen if self._minlen is not None else 1
            maxlen = self._maxlen if self._maxlen is not None else N
            stride = self._stride if self._stride is not None else 1
            for i in range(0, N - maxlen, stride):
                for j in range(i + minlen, min(i + maxlen, N), stride):
                    yield (traj.index(lambda t: t[i:j]), 0)


class SpatialRelation(Node):
    def __init__(self, obj1, obj2):
        self.obj1 = obj1
        self.obj2 = obj2
        self.children = [obj1, obj2]

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
