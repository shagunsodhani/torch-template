"""Implements a serializable Box"""
from box import Box, _to_json


class SerializableConfig(Box):
    """serializable box"""

    def to_json(self, filename=None,
                encoding="utf-8", errors="strict", **json_kwargs):
        """
        Transform the Box object into a JSON string.

        :param filename: If provided will save to file
        :param encoding: File encoding
        :param errors: How to handle encoding errors
        :param json_kwargs: additional arguments to pass to json.dump(s)
        :return: string of JSON or return of `json.dump`
        """
        _dict = self.to_dict()
        _dict['general']['device'] = _dict['general']['device'].type
        for key in ["observation_space", "action_space"]:
            _dict["env"][key] = str(_dict['env'][key])
        return _to_json(_dict, filename=filename,
                        encoding=encoding, errors=errors, **json_kwargs)


def _get_config_box(_dict, frozen_box=False):
    """Wrapper to get a box"""
    return SerializableConfig(
        _dict,
        default_box_attr=None,
        box_duplicates="ignore",
        frozen_box=frozen_box
    )


def get_config_box(_dict):
    """Wrapper to get a box"""
    return _get_config_box(_dict, frozen_box=False)


def get_forzen_config_box(_dict):
    """Wrapper to get a frozen box"""
    return _get_config_box(_dict, frozen_box=True)
