"""Dataset recording (LeRobotDataset) for so101_tool.

Importing this package is safe without lerobot installed; the lerobot
dependency is only required when a :class:`DatasetRecorder` is instantiated.
"""

from so101_tool.data.recorder import DatasetRecorder

__all__ = ["DatasetRecorder"]
