# Unit tests for contrib/e_means/device_profile.py. No GPU needed: devices are mocked.

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from faiss.contrib.e_means import device_profile as dp


class ArchFamilyTest(unittest.TestCase):

    def test_target_families(self):
        for arch, family in (("gfx942", "cdna3"), ("gfx950", "cdna4"),
                             ("gfx1100", "rdna3"), ("gfx1200", "rdna4")):
            with self.subTest(arch=arch):
                self.assertEqual(dp.family_of(arch), family)

    def test_siblings_match_by_family(self):
        for arch, family in (("gfx940", "cdna3"), ("gfx941", "cdna3"),
                             ("gfx1101", "rdna3"), ("gfx1102", "rdna3"),
                             ("gfx1151", "rdna3"), ("gfx1201", "rdna4")):
            with self.subTest(arch=arch):
                self.assertEqual(dp.family_of(arch), family)

    def test_non_target_archs(self):
        for arch in ("gfx90a", "gfx908", "gfx1030", "gfx1250", "gfx951", "", None):
            with self.subTest(arch=arch):
                self.assertIsNone(dp.family_of(arch))

    def test_feature_suffix_and_case(self):
        self.assertEqual(dp.normalize_arch("gfx942:sramecc+:xnack-"), "gfx942")
        self.assertEqual(dp.family_of("GFX1200"), "rdna4")
        self.assertEqual(dp.family_of("gfx90a:sramecc+:xnack-"), None)

    def test_profile_facts(self):
        p = dp.profile_for_arch("gfx942:sramecc+:xnack-")
        self.assertEqual((p.vendor, p.arch, p.family, p.wavefront), ("amd", "gfx942", "cdna3", 64))
        self.assertTrue(p.fp32_matrix)
        self.assertFalse(p.native_fp8)       # FNUZ only; E4M3FN is emulated
        self.assertTrue(p.lds_confirmed)
        self.assertEqual(p.lds_bytes, 64 * 1024)
        self.assertTrue(p.known_amd)

        rx = dp.profile_for_arch("gfx1201")
        self.assertEqual((rx.family, rx.wavefront), ("rdna4", 32))
        self.assertFalse(rx.fp32_matrix)
        self.assertTrue(rx.native_fp8)
        self.assertTrue(rx.lds_confirmed)        # measured on gfx1200
        self.assertFalse(dp.profile_for_arch("gfx950").lds_confirmed)
        self.assertFalse(dp.profile_for_arch("gfx1100").lds_confirmed)

        self.assertTrue(dp.profile_for_arch("gfx950").native_fp8)
        self.assertFalse(dp.profile_for_arch("gfx1100").native_fp8)
        self.assertFalse(dp.profile_for_arch("gfx1100").fp32_matrix)

    def test_unknown_amd_profile(self):
        p = dp.profile_for_arch("gfx90a")
        self.assertTrue(p.is_amd)
        self.assertFalse(p.known_amd)
        self.assertIsNone(p.family)
        self.assertIsNone(p.lds_bytes)


class DeviceLookupTest(unittest.TestCase):

    def setUp(self):
        dp._amd_arch.cache_clear()

    def tearDown(self):
        dp._amd_arch.cache_clear()

    def test_cpu(self):
        p = dp.device_profile("cpu")
        self.assertEqual(p.vendor, "cpu")
        with mock.patch.object(torch.version, "hip", "7.16"):
            self.assertIsNone(dp.nvidia_capability("cpu"))

    def test_nvidia_capability_mirrors_upstream_call(self):
        # Upstream called get_device_capability(x.device) with no device-type check;
        # tests rely on that with CPU tensors and a mocked capability.
        with mock.patch.object(torch.version, "hip", None), \
             mock.patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)) as get:
            self.assertEqual(dp.nvidia_capability(torch.device("cpu")), (9, 0))
        get.assert_called_once_with(torch.device("cpu"))

    def test_nvidia(self):
        with mock.patch.object(torch.version, "hip", None), \
             mock.patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            self.assertEqual(dp.nvidia_capability("cuda:0"), (9, 0))
            p = dp.device_profile("cuda:0")
        self.assertEqual((p.vendor, p.arch, p.capability), ("nvidia", "sm90", (9, 0)))
        self.assertFalse(p.is_amd)

    def test_nvidia_capability_is_not_cached(self):
        with mock.patch.object(torch.version, "hip", None):
            with mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)):
                self.assertEqual(dp.nvidia_capability("cuda:0"), (8, 0))
            with mock.patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
                self.assertEqual(dp.nvidia_capability("cuda:0"), (9, 0))

    def test_amd_never_reports_nvidia_capability(self):
        capability = mock.Mock(return_value=(9, 0))
        with mock.patch.object(torch.version, "hip", "7.16"), \
             mock.patch.object(torch.cuda, "get_device_capability", capability):
            self.assertIsNone(dp.nvidia_capability("cuda:0"))
        capability.assert_not_called()

    def test_amd_profile_from_gcn_arch_name(self):
        props = SimpleNamespace(gcnArchName="gfx1200")
        with mock.patch.object(torch.version, "hip", "7.16"), \
             mock.patch.object(torch.cuda, "get_device_properties", return_value=props):
            p = dp.device_profile("cuda:0")
        self.assertEqual((p.vendor, p.arch, p.family), ("amd", "gfx1200", "rdna4"))

    def test_default_cuda_index_uses_current_device(self):
        props = SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-")
        with mock.patch.object(torch.version, "hip", "7.16"), \
             mock.patch.object(torch.cuda, "current_device", return_value=0), \
             mock.patch.object(torch.cuda, "get_device_properties", return_value=props) as get:
            p = dp.device_profile("cuda")
            dp.device_profile(torch.device("cuda", 0))
        self.assertEqual(p.family, "cdna3")
        get.assert_called_once_with(0)     # cached per device index


if __name__ == "__main__":
    unittest.main()
