"""Global innovation number tracking for NEAT-style crossover alignment.

Every structural event (new connection, new node) gets a unique, permanent
innovation number. During crossover, genes with the same innovation number
represent the same historical structural feature and can be aligned correctly.
"""
from __future__ import annotations


class InnovationTracker:
    """Maps structural events to unique, monotonically increasing innovation numbers.

    Connections: keyed by (src_innovation, tgt_innovation) so the same logical
    connection is always assigned the same number, even when added independently
    in different genomes (convergent evolution).

    Node splits: keyed by the innovation number of the split connection so the
    same split always produces the same node innovation and the same two
    replacement-connection innovations.
    """

    def __init__(self) -> None:
        self._counter: int = 0
        self._conn_map: dict[tuple[int, int], int] = {}
        self._split_map: dict[int, tuple[int, int, int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next(self) -> int:
        n = self._counter
        self._counter += 1
        return n

    def get_connection(self, src_innov: int, tgt_innov: int) -> int:
        """Return (creating if necessary) the innovation number for a connection."""
        key = (src_innov, tgt_innov)
        if key not in self._conn_map:
            self._conn_map[key] = self.next()
        return self._conn_map[key]

    def get_split(self, conn_innov: int) -> tuple[int, int, int]:
        """Return (node_innov, conn_in_innov, conn_out_innov) for splitting a connection.

        When the same connection (by innovation number) is split in different
        genomes, the resulting node and two new connections always get the
        same innovation numbers.  Identical splits in parallel lineages are
        therefore recognised as the same structural event during crossover.
        """
        if conn_innov not in self._split_map:
            node_innov  = self.next()
            conn_in     = self.next()
            conn_out    = self.next()
            self._split_map[conn_innov] = (node_innov, conn_in, conn_out)
        return self._split_map[conn_innov]
