"""HAHyPO algorithm.

HAHyPO keeps the HAPPO clipped policy update, while its runner mixes the
standard GAE advantage with a group-relative trajectory advantage inspired by
GRPO. The actor class intentionally inherits HAPPO so the policy update itself
stays comparable.
"""

from harl.algorithms.actors.happo import HAPPO


class HAHyPO(HAPPO):
    """Hybrid-advantage HAPPO."""

    pass
