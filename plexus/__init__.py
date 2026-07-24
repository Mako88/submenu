"""plexus -- a bio-inspired neural model built for locality, not for GPUs.

The design discipline is a single rule: no operation may require globally
synchronised state. Biology obeys that rule, which is why a brain tolerates
hundreds of milliseconds of delay and degrades gracefully when parts of it go
missing. Backpropagation violates it, which is why deep networks need tightly
coupled hardware.

Following the rule -- rather than following biology's chemistry -- lets us keep
what neurons compute while dropping what neurons merely had to cope with.
"""

from .column import Column, ColumnConfig
from .events import Event, EventBuffer
from .network import Plexus, TrainReport
from .readout import LinearReadout
from .tasks import DelayedXOR, Episode, TemporalPatterns
from .transport import LocalTransport, Transport

__all__ = [
    "Column",
    "ColumnConfig",
    "Event",
    "EventBuffer",
    "LinearReadout",
    "LocalTransport",
    "Plexus",
    "Transport",
    "TrainReport",
    "DelayedXOR",
    "Episode",
    "TemporalPatterns",
]
