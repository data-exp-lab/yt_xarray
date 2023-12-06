from typing import List, Optional, Union

import xarray as xr
import yt


def _get_default_ds(ds_xr: xr.Dataset, field):
    geom = ds_xr.yt.geometry  # will trigger inference here

    # other load_grid options
    # use_callable: bool = True,
    # sel_dict: Optional[dict] = None,  dont need this one
    # sel_dict_type: Optional[str] = "isel",  dont need this one
    # chunksizes: Optional[int] = None,

    # if the grid were cached this might be easier...
    ds = ds_xr.yt.load_grid(fields=field, geometry=geom)
    return ds


def _yt_2D_plot(yt_function, ds_xr: xr.Dataset, normal, field, **im_kwargs):
    ds = _get_default_ds(ds_xr, field)
    # normal = validate_normal(normal)
    return yt_function(ds, normal, ("stream", field), **im_kwargs)


class YTVisContainer:
    def __init__(self, xr_obj):
        self._obj = xr_obj

    def SlicePlot(self, normal, field, **im_kwargs):
        return _yt_2D_plot(yt.SlicePlot, self._obj, normal, field, **im_kwargs)

    def ProjectionPlot(self, normal, field, **im_kwargs):
        return _yt_2D_plot(yt.ProjectionPlot, self._obj, normal, field, **im_kwargs)

    def PhasePlot(
        self,
        x_field: str,
        y_field: str,
        z_fields: Union[str, List[str]],
        weight_field: Optional[str] = None,
        x_bins: Optional[int] = 128,
        y_bins: Optional[int] = 128,
        accumulation: Optional[bool] = False,
        fractional: Optional[bool] = False,
        fontsize: Optional[int] = 18,
        figure_size: Optional[int] = 8.0,
        shading: Optional[str] = "nearest",
    ):
        """
        x_field : str
            The x binning field for the profile.
        y_field : str
            The y binning field for the profile.
        z_fields : str or list
            The field or fields to be profiled.
        """
        if isinstance(z_fields, str):
            z_fields = [
                z_fields,
            ]

        fields_needed = list(set([x_field, y_field] + z_fields))
        ds = _get_default_ds(self._obj, fields_needed)

        x_field = ("stream", x_field)
        y_field = ("stream", y_field)
        z_fields = [("stream", z_f) for z_f in z_fields]

        return yt.PhasePlot(
            ds,
            x_field,
            y_field,
            z_fields,
            weight_field=weight_field,
            x_bins=x_bins,
            y_bins=y_bins,
            accumulation=accumulation,
            fractional=fractional,
            fontsize=fontsize,
            figure_size=figure_size,
            shading=shading,
        )

    def ProfilePlot(
        self,
        x_field,
        y_fields,
        weight_field=None,
        n_bins=64,
        accumulation=False,
        fractional=False,
        label=None,
        plot_spec=None,
        x_log=True,
        y_log=True,
    ):
        fields_needed = list(set([x_field, y_fields]))

        if weight_field is not None and weight_field not in fields_needed:
            fields_needed.append(weight_field)

        ds = _get_default_ds(self._obj, fields_needed)
        ad = ds.all_data()

        return yt.ProfilePlot(
            ad,
            x_field,
            y_fields,
            weight_field=weight_field,
            n_bins=n_bins,
            accumulation=accumulation,
            fractional=fractional,
            label=label,
            plot_spec=plot_spec,
            x_log=x_log,
            y_log=y_log,
        )
