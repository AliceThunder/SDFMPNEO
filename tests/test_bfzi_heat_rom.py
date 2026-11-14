import numpy as np

from sdfmpneo_bfzi.heat_rom import HeatROM


def test_heat_rom_exact_small_system(tmp_path):
    cap = np.array([2., 3., 4.])
    h = np.array([[2., -1., -1.], [-1., 2., -1.], [-1., -1., 2.]])
    snaps = np.array([[294., 293., 293.], [293., 295., 293.],
                      [293., 293., 296.]])
    rom = HeatROM(cap, h, snaps, rank=3, ambient=293.)
    assert rom.rank == 3
    t = np.array([310., 302., 298.])
    np.testing.assert_allclose(rom.decode(rom.encode(t)), t, atol=1e-10)
    power = np.array([1., 2., 3.])
    dt = 2.
    z = rom.step(rom.encode(t), power, dt)
    expected = np.linalg.solve(np.diag(cap) + dt*h, cap*(t-293.)+dt*power)+293.
    np.testing.assert_allclose(rom.decode(z), expected, atol=1e-10)
    path = tmp_path / "rom.npz"
    rom.save(path)
    np.testing.assert_allclose(HeatROM.load(path).decode(z), rom.decode(z))
