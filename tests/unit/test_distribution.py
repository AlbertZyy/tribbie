from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from tribbie.distribution import (
    DistributionError,
    DistributionLayout,
    DistributionVersion,
    DuplicateGlobalIdError,
    IndexDistribution,
    InvalidGlobalIdError,
    InvalidIndexError,
    InvalidOwnerError,
    RankMismatchError,
    UnsupportedLayoutError,
    make_halo_plan,
)


def _distribution(**overrides):
    kwargs = dict(
        global_ids=np.array([100, 7, 200, 9], dtype=np.int64),
        owners=np.array([0, 0, 1, 1], dtype=np.int32),
        global_size=1000,
        self_rank=0,
        version=DistributionVersion(topology=1, numbering=2, ghost_layout=3),
    )
    kwargs.update(overrides)
    return IndexDistribution(**kwargs)


def test_version_is_frozen_with_value_semantics():
    a = DistributionVersion(topology=1, numbering=2, ghost_layout=3)
    b = DistributionVersion(topology=1, numbering=2, ghost_layout=3)
    c = DistributionVersion(topology=1, numbering=2)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert DistributionVersion.zero() == DistributionVersion()
    with pytest.raises(AttributeError):
        a.topology = 9


def test_public_error_types_share_distribution_error():
    assert issubclass(InvalidGlobalIdError, DistributionError)
    assert issubclass(DuplicateGlobalIdError, DistributionError)
    assert issubclass(InvalidOwnerError, DistributionError)
    assert issubclass(InvalidIndexError, DistributionError)
    assert issubclass(RankMismatchError, DistributionError)
    assert issubclass(UnsupportedLayoutError, DistributionError)


def test_general_layout_classifies_owned_and_ghost_from_owners():
    dist = _distribution()
    assert dist.layout is DistributionLayout.GENERAL
    assert dist.is_owned_first is False
    assert dist.self_rank == 0
    assert dist.global_size == 1000
    assert dist.local_size == 4
    assert dist.owned_size == 2
    assert dist.ghost_size == 2

    assert dist.is_owned(0) and dist.is_owned(1)
    assert dist.is_ghost(2) and dist.is_ghost(3)
    assert not dist.is_owned(2)
    assert not dist.is_ghost(0)


def test_query_methods_local_to_global_owner_and_inverse():
    dist = _distribution()
    assert dist.local_to_global(0) == 100
    assert dist.local_to_global(3) == 9
    assert dist.owner(0) == 0
    assert dist.owner(2) == 1
    assert dist.global_to_local(100) == 0
    assert dist.global_to_local(9) == 3
    assert dist.global_to_local(404) is None


def test_query_out_of_range_raises_invalid_index():
    dist = _distribution()
    for i in (-1, 4, 100):
        with pytest.raises(InvalidIndexError):
            dist.is_owned(i)
        with pytest.raises(InvalidIndexError):
            dist.is_ghost(i)
        with pytest.raises(InvalidIndexError):
            dist.local_to_global(i)
        with pytest.raises(InvalidIndexError):
            dist.owner(i)


def test_derived_owned_and_ghost_views():
    dist = _distribution()
    assert np.array_equal(dist.owned_global_ids, [100, 7])
    assert np.array_equal(dist.ghost_global_ids, [200, 9])
    assert np.array_equal(dist.ghost_owners, [1, 1])


def test_input_validation_errors():
    with pytest.raises(InvalidGlobalIdError):
        _distribution(global_ids=np.array([100, 7, -1, 9], dtype=np.int64))
    with pytest.raises(InvalidGlobalIdError):
        _distribution(global_ids=np.array([100, 7, 1000, 9], dtype=np.int64))
    with pytest.raises(InvalidGlobalIdError):
        _distribution(global_ids=np.array([100, 7, 200, 9], dtype=np.int32))
    with pytest.raises(DuplicateGlobalIdError):
        _distribution(global_ids=np.array([100, 7, 100, 9], dtype=np.int64))
    with pytest.raises(InvalidOwnerError):
        _distribution(owners=np.array([0, 0, 1], dtype=np.int32))
    with pytest.raises(InvalidOwnerError):
        _distribution(owners=np.array([0, 0, -1, 1], dtype=np.int32))
    with pytest.raises(InvalidOwnerError):
        _distribution(owners=np.array([0.0, 0.0, 1.0, 1.0]))
    with pytest.raises(InvalidGlobalIdError):
        _distribution(global_size=-1)
    with pytest.raises(InvalidOwnerError):
        _distribution(self_rank=-2)


def test_metadata_arrays_are_read_only_and_copied():
    ids = np.array([100, 7, 200, 9], dtype=np.int64)
    owners = np.array([0, 0, 1, 1], dtype=np.int32)
    dist = IndexDistribution(ids, owners, global_size=1000, self_rank=0)

    with pytest.raises(ValueError):
        dist.global_ids[0] = 999
    with pytest.raises(ValueError):
        dist.owned_global_ids[0] = 999

    ids[0] = 555
    owners[2] = 0
    assert dist.local_to_global(0) == 100
    assert dist.owner(2) == 1


def test_empty_distribution():
    dist = IndexDistribution(
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int32),
        global_size=0,
        self_rank=0,
    )
    assert dist.local_size == 0
    assert dist.owned_size == 0
    assert dist.ghost_size == 0
    assert dist.global_to_local(5) is None


def test_serialization_round_trip():
    dist = _distribution()
    data = dist.to_dict()
    restored = IndexDistribution.from_dict(data)

    assert restored.self_rank == dist.self_rank
    assert restored.global_size == dist.global_size
    assert restored.version == dist.version
    assert restored.layout is dist.layout
    assert np.array_equal(restored.global_ids, dist.global_ids)
    assert np.array_equal(restored.owners, dist.owners)

    data["global_ids"][0] = 999
    assert dist.local_to_global(0) == 100


def test_reserved_owned_first_ghost_sorted_layout():
    with pytest.raises(UnsupportedLayoutError):
        IndexDistribution(
            np.array([100, 7], dtype=np.int64),
            np.array([0, 0], dtype=np.int32),
            global_size=1000,
            self_rank=0,
            layout=DistributionLayout.OWNED_FIRST_GHOST_SORTED,
        )
    with pytest.raises(UnsupportedLayoutError):
        IndexDistribution.from_owned_first_ghost_sorted(
            np.array([100, 7], dtype=np.int64),
            np.array([0, 0], dtype=np.int32),
            global_size=1000,
            self_rank=0,
        )


def test_make_halo_plan_two_way_empty_on_single_rank():
    dist = IndexDistribution(
        np.array([100, 7], dtype=np.int64),
        np.array([0, 0], dtype=np.int32),
        global_size=1000,
        self_rank=0,
    )
    reduce_plan, broadcast_plan = make_halo_plan(dist, MPI.COMM_SELF, direction="two_way")
    assert reduce_plan.neighbors == ()
    assert broadcast_plan.neighbors == ()


def test_make_halo_plan_single_direction_empty_on_single_rank():
    dist = IndexDistribution(
        np.array([100, 7], dtype=np.int64),
        np.array([0, 0], dtype=np.int32),
        global_size=1000,
        self_rank=0,
    )
    assert make_halo_plan(dist, MPI.COMM_SELF, direction="owner_to_ghost").neighbors == ()
    assert make_halo_plan(dist, MPI.COMM_SELF, direction="ghost_to_owner").neighbors == ()


def test_make_halo_plan_rejects_self_rank_mismatch():
    dist = _distribution(self_rank=1)
    with pytest.raises(RankMismatchError):
        make_halo_plan(dist, MPI.COMM_SELF, direction="two_way")


def test_distribution_plan_two_way_method_delegates():
    dist = IndexDistribution(
        np.array([100, 7], dtype=np.int64),
        np.array([0, 0], dtype=np.int32),
        global_size=1000,
        self_rank=0,
    )
    reduce_plan, broadcast_plan = dist.plan_two_way(MPI.COMM_SELF)
    assert reduce_plan.neighbors == ()
    assert broadcast_plan.neighbors == ()
