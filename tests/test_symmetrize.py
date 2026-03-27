# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# moLDGA — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#          Eliashberg Equation Solver for Strongly Correlated Electron Systems

import pytest

from moldga.symmetrize_new import *


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_index2component_general_and_back(num_bands):
    for ind in range(1, 16 * num_bands**4 + 1):
        bandspin, band, spin = index2component_general(num_bands, 4, ind)
        ind_back = component2index_general(num_bands, list(band), list(spin))
        assert ind_back == ind


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_index2component_general_and_back_raises_if_index_too_large_or_too_small(num_bands):
    with pytest.raises(ValueError):
        bandspin, band, spin = index2component_general(num_bands, 4, 16 * num_bands**4 + 1)
        _ = component2index_general(num_bands, list(band), list(spin))

    with pytest.raises(ValueError):
        bandspin, band, spin = index2component_general(num_bands, 4, 0)
        _ = component2index_general(num_bands, list(band), list(spin))


def test_component2index_general_invalid_num_bands():
    with pytest.raises(AssertionError):
        component2index_general(0, [0], [0])


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_index2component_band_and_back(num_bands):
    orbs = list(it.product(range(num_bands), repeat=4))

    for orb in orbs:
        ind = component2index_band(num_bands, 4, list(orb))
        indices_back = index2component_band(num_bands, 4, ind)
        assert indices_back == list(orb)


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components(num_bands):
    result = get_worm_components(num_bands)
    if num_bands == 1:
        assert result == [1, 4, 7, 10, 13, 16]
    assert len(result) == 6 * num_bands**4
