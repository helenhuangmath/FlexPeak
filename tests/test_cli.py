from types import SimpleNamespace

from flexpeak import cli


class _PeakSet:
    def __len__(self):
        return 1


def test_qc_with_existing_peaks_does_not_recall(monkeypatch, capsys):
    treatment = object()
    peaks = _PeakSet()

    args = SimpleNamespace(
        chroms=None,
        preset=None,
        bin_size=None,
        qvalue=0.05,
        min_width=None,
        min_fold=None,
        no_subpeaks=False,
        poisson=False,
        background_exclude_fold=None,
        no_dwell_floor=False,
        segment_bin_size=None,
        treatment="signal.bw",
        control=None,
        mapq=30,
        no_cache=True,
        cache_dir=None,
        peaks="calls.nested.tsv",
        json=True,
    )

    monkeypatch.setattr(cli, "_load", lambda *a, **k: treatment)

    import flexpeak.caller
    import flexpeak.peaks
    import flexpeak.qc.metrics

    def fail_call_peaks(*args, **kwargs):
        raise AssertionError("call_peaks should not run when --peaks is supplied")

    def fake_qc_metrics(observed_treatment, observed_peaks, control, background=None):
        assert observed_treatment is treatment
        assert observed_peaks is peaks
        assert control is None
        assert background is None
        return {"n_regions": len(observed_peaks)}

    monkeypatch.setattr(flexpeak.caller, "call_peaks", fail_call_peaks)
    monkeypatch.setattr(flexpeak.peaks, "read_nested_tsv", lambda path: peaks)
    monkeypatch.setattr(flexpeak.qc.metrics, "qc_metrics", fake_qc_metrics)

    assert cli.cmd_qc(args) == 0
    assert '"n_regions": 1' in capsys.readouterr().out
