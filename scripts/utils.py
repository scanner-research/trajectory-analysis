import math
from vgrid import IntervalBlock, NamedIntervalSet
from esper.widget import vgrid_widget

from pyro import poutine
from pyro.contrib.autoguide import AutoGuide
from pyro.infer import config_enumerate
from pyro.poutine.util import prune_subsample_sites
import pyro.distributions as dist
import pyro


def img_grid(imgs, columns=3):
    import matplotlib.pyplot as plt
    max_width = 900
    rows = math.ceil(len(imgs) / columns)
    fig = plt.figure(figsize=(15, rows * 3))
    for i, img in enumerate(imgs):
        fig.add_subplot(rows, columns, i + 1)
        plt_img = plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()


def vgrid_traj(trajectories):
    return vgrid_widget(
        video_meta=[t.obj.video.for_vgrid() for t in trajectories],
        interval_blocks=[
            IntervalBlock(video_id=t.obj.video.id,
                          interval_sets=[
                              NamedIntervalSet(name='default',
                                               interval_set=t.for_vgrid())
                          ]) for t in trajectories
        ])


def make_batch(signal, framerate):
    for i in range(0, len(signal), framerate):
        yield signal[i:i + framerate]


# Adapted from AutoDiscreteParallel
class AutoDiscreteSequential(AutoGuide):
    """
    A discrete mean-field guide that learns a latent discrete distribution for
    each discrete site in the model.
    """

    def _setup_prototype(self, *args, **kwargs):
        # run the model so we can inspect its structure
        model = config_enumerate(self.model)
        self.prototype_trace = poutine.block(poutine.trace(model).get_trace)(
            *args, **kwargs)
        self.prototype_trace = prune_subsample_sites(self.prototype_trace)
        if self.master is not None:
            self.master()._check_prototype(self.prototype_trace)

        self._discrete_sites = []
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            if site["infer"].get("enumerate") != "sequential":
                raise NotImplementedError(
                    'Expected sample site "{}" to be discrete and '
                    'configured for sequential enumeration'.format(name))

            # collect discrete sample sites
            fn = site["fn"]
            Dist = type(fn)
            if Dist in (dist.Bernoulli, dist.Categorical,
                        dist.OneHotCategorical):
                params = [("probs", fn.probs.detach().clone(),
                           fn.arg_constraints["probs"])]
            else:
                raise NotImplementedError("{} is not supported".format(
                    Dist.__name__))
            self._discrete_sites.append((site, Dist, params))

    def __call__(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        # enumerate discrete latent samples
        result = {}
        for site, Dist, param_spec in self._discrete_sites:
            name = site["name"]
            dist_params = {
                param_name:
                pyro.param("{}_{}_{}".format(self.prefix, name, param_name),
                           param_init,
                           constraint=param_constraint)
                for param_name, param_init, param_constraint in param_spec
            }
            discrete_dist = Dist(**dist_params)

            result[name] = pyro.sample(name,
                                       discrete_dist,
                                       infer={"enumerate": "sequential"})

        return result
