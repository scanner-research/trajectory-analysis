from esper.models import Video, EsperModel, CharField, FK
from django.db.models import FloatField, IntegerField
import cv2
import numpy as np
from rekall import Interval, Bounds3D, IntervalSet
from copy import copy

# Put your custom models in here. Run python3 manage.py makemigrations && python3 manage.py migrate to update
# the SQL database. For more detail: https://docs.djangoproject.com/en/2.2/topics/migrations/

# Also take a look at esper.models for some useful base classes you can mix-in to your models.


class Video(Video):
    pixel2meter = FloatField()


class Pedestrian(EsperModel):
    video = FK(Video)

    def trajectory(self):
        if not hasattr(self, '_traj'):
            self._traj = PedTraj(self)
        return self._traj


class Pos:
    def to_pixel(self):
        obj = self.ped if isinstance(self, PedPos) else self.veh
        p2m = obj.video.pixel2meter
        return (int(p2m * self.x), int(p2m * self.y))

    def to_pixel_norm(self):
        obj = self.ped if isinstance(self, PedPos) else self.veh
        (x, y) = self.to_pixel()
        return (x / obj.video.width, y / obj.video.height)

    def time(self):
        return self.frame

    def for_numpy(self):
        return np.array([self.x, self.y])


class PedPos(Pos, EsperModel):
    ped = FK(Pedestrian)
    frame = IntegerField()
    x = FloatField()
    y = FloatField()
    vx = FloatField()
    vy = FloatField()


class Vehicle(EsperModel):
    video = FK(Video)

    def trajectory(self):
        return VehTraj(self)


class VehPos(Pos, EsperModel):
    veh = FK(Vehicle)
    frame = IntegerField()
    x = FloatField()
    y = FloatField()
    psi = FloatField()
    vel = FloatField()


class Trajectory:
    def draw(self, img=None, color=(255, 0, 0)):
        if img is None:
            [img] = self.obj.video.for_hwang().retrieve([self.pos[0].frame])
        for p in self.pos:
            cv2.circle(img, p.to_pixel(), 3, color, -1)
        return img

    def for_numpy(self):
        return np.array([[p.x, p.y] for p in self.pos])

    def for_vgrid(self):
        fps = self.obj.video.fps
        pad = 0.02
        return IntervalSet([
            Interval(
                Bounds3D(t1=p.frame / fps,
                         t2=(p.frame + 1) / fps,
                         x1=p.to_pixel_norm()[0] - pad,
                         x2=p.to_pixel_norm()[0] + pad,
                         y1=p.to_pixel_norm()[1] - pad,
                         y2=p.to_pixel_norm()[1] + pad)) for p in self.pos
        ])

    def velocity(self, stride=1):
        p = self.for_numpy()[::stride]
        return p[:-1] - p[1:]

    def frame(self, i):
        if i in self._pos_index:
            return self._pos_index[i]
        else:
            return None

    def index(self, f):
        t = copy(self)
        t.pos = f(t.pos)
        return t

    def build_index(self):
        self._pos_index = {p.frame: p for p in self.pos}


class VehTraj(Trajectory):
    def __init__(self, veh, spark=None):
        if spark is not None:
            pos_df = spark.load(VehPos)
            self.pos = pos_df.where(pos_df.veh_id == veh.id).orderBy(
                pos_df.frame).collect()
        else:
            self.pos = list(
                VehPos.objects.filter(veh=veh).select_related(
                    'veh', 'veh__video').order_by('frame'))
        self.obj = veh
        self.build_index()


class PedTraj(Trajectory):
    def __init__(self, ped, spark=None):
        if spark is not None:
            pos_df = spark.load(PedPos)
            self.pos = pos_df.where(pos_df.ped_id == ped.id).orderBy(
                pos_df.frame).collect()
        else:
            self.pos = list(
                PedPos.objects.filter(ped=ped).select_related(
                    'ped', 'ped__video').order_by('frame'))
        self.obj = ped
        self.build_index()
