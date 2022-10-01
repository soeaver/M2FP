import wandb
import os
from detectron2.utils.events import EventWriter, get_event_storage


class WandBWriter(EventWriter):
    """
    Write all scalars to a WandB.
    """

    def __init__(self, cfg, window_size=20, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self.cfg = cfg
        wandb.init(
            entity=cfg.WANDB.ENTITY,
            name=cfg.WANDB.NAME if cfg.WANDB.NAME != "" else os.path.basename(cfg.OUTPUT_DIR.strip('/')),
            project=cfg.WANDB.PROJECT,
            config=cfg
        )
        self._window_size = window_size
        self._last_write = -1

    def write(self):
        storage = get_event_storage()
        new_last_write = self._last_write
        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            if iter > self._last_write:
                wandb.log({k: v}, step=iter)
                new_last_write = max(new_last_write, iter)
        self._last_write = new_last_write

        # TODO: log image
        # # storage.put_{image,histogram} is only meant to be used by
        # # tensorboard writer. So we access its internal fields directly from here.
        # if len(storage._vis_data) >= 1:
        #     for img_name, img, step_num in storage._vis_data:
        #         self._writer.add_image(img_name, img, step_num)
        #     # Storage stores all image data and rely on this writer to clear them.
        #     # As a result it assumes only one writer will use its image data.
        #     # An alternative design is to let storage store limited recent
        #     # data (e.g. only the most recent image) that all writers can access.
        #     # In that case a writer may not see all image data if its period is long.
        #     storage.clear_images()
        #
        # if len(storage._histograms) >= 1:
        #     for params in storage._histograms:
        #         self._writer.add_histogram_raw(**params)
        #     storage.clear_histograms()
