import logging

ytxr_log = logging.getLogger("yt_xarray")
ytxr_log.setLevel(logging.INFO)

_formatter = logging.Formatter("%(name)s : [%(levelname)s ] %(asctime)s:  %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(_formatter)
ytxr_log.addHandler(stream_handler)
