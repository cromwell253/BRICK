import numpy as np
import torch

from . import TemporalDataset, SpatioTemporalDataset


class ImputationDataset(TemporalDataset):

    def __init__(self, data,
                 index=None,
                 mask=None,
                 eval_mask=None,
                 freq=None,
                 trend=None,
                 scaler=None,
                 window=24,
                 stride=1,
                 exogenous=None):
        # data: (34272, 207)
        # index: (34272, )
        # mask: (34272, 207)
        # eval_mask: (34272, 207)
        # freq: None
        # trend: None
        # scaler: None
        # window: 24
        # stride: 1
        # exogenous: None
        if mask is None:  # ignore
            mask = np.ones_like(data)
        if exogenous is None:  # run
            exogenous = dict()
        exogenous['mask_window'] = mask
        if eval_mask is not None:
            exogenous['eval_mask_window'] = eval_mask
        super(ImputationDataset, self).__init__(data,
                                                index=index,
                                                exogenous=exogenous,
                                                trend=trend,
                                                scaler=scaler,
                                                freq=freq,
                                                window=window,
                                                horizon=window,
                                                delay=-window,
                                                stride=stride)

    def get(self, item, preprocess=False):
        # preprocess: true
        res, transform = super(ImputationDataset, self).get(item, preprocess)
        # res['x']: (24, 207) - unmasked
        # res['mask']: (24, 207)
        # transform['scale']: (1, 207)
        # transform['bias']: (1, 207)
        res['x'] = torch.where(res['mask'], res['x'], torch.zeros_like(res['x']))
        return res, transform


class GraphImputationDataset(ImputationDataset, SpatioTemporalDataset):
    pass


class ForecastImputationDataset(ImputationDataset):
    """
    Dataset providing both imputation targets (same window) and
    forecasting targets (future H timesteps after the window).
    """

    def __init__(self, data, index=None, forecast_horizon=12, **kwargs):
        self._forecast_horizon = forecast_horizon
        super().__init__(data, index=index, **kwargs)

    @property
    def sample_span(self):
        base_span = max(self.horizon_offset + self.horizon, self.window)
        return base_span + self._forecast_horizon

    def get(self, item, preprocess=False):
        res, transform = super().get(item, preprocess)
        idx = self._indices[item]
        forecast_start = idx + self.window
        forecast_end = forecast_start + self._forecast_horizon

        y_forecast = self.data[forecast_start:forecast_end]

        if hasattr(self, 'mask') and self.mask is not None:
            res['forecast_mask'] = self.mask[forecast_start:forecast_end]

        # NOTE: do NOT scale y_forecast here. Like y, it stays in original space.
        # The filler's training_step will handle scaling via _preprocess when needed
        # (scaled_target=True), consistent with how y is handled.
        res['y_forecast'] = y_forecast
        return res, transform


class GraphForecastImputationDataset(ForecastImputationDataset, SpatioTemporalDataset):
    pass
