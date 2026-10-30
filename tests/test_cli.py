"""CLI paths, measured timing calls, and transactional dataset download behavior."""

import hashlib
import io
from pathlib import Path
import subprocess
import sys
from urllib.error import URLError

import pytest

from qrc_pulsar_bench import cli, datasets

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('arguments', [
    ['simulate', '--orbits', '1', '--points', '30'],
    ['compare', '--orbits', '1', '--points', '30'],
    ['list'], ['all'], ['formula-cost', '--repeats', '2'],
    ['run', '--system', 'PSR_J0337+1715'], ['run', '--system', 'PSR_J0737-3039'],
    ['run', '--system', 'PSR_B1913+16'], ['run', '--system', 'PSR_J1738+0333'],
    ['run', '--system', 'Kepler-16b'],
    ['custom', '--path', str(ROOT/'data/nanograv/J1614-2230_NANOGrav_12yv4.all.res')],
])
def test_cli_smoke(arguments):
    result = subprocess.run([sys.executable, str(ROOT/'run_benchmarks.py'), *arguments],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout
    assert 'Zero Phase Slip' not in result.stdout
    assert 'CONFIRMED' not in result.stdout


def test_compare_executes_real_calls(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, '_solver', lambda system: lambda s: calls.append(s.name))
    cli.main(['formula-cost', '--repeats', '3'])
    assert len(calls) == len(cli.SYSTEMS)*4  # One warmup and three measurements.
    output = capsys.readouterr().out
    assert '3 evaluations' in output
    assert '12.4 seconds' not in output


def test_zenodo_cli_accepts_existing_small_fixture_without_overwriting(tmp_path, monkeypatch, capsys):
    path = tmp_path/'residuals.csv'
    content = ('# MJD (approx), residual, uncertainty (phase), phase, delta\n'
               '58000,0.001,0.001,1,1\n58001,-0.001,0.001,1,-1\n58002,0,0.001,1,0\n')
    path.write_text(content)
    def no_network(*args, **kwargs):
        pytest.fail('Existing data must not trigger a download')
    monkeypatch.setattr(datasets.urllib.request, 'urlopen', no_network)
    cli.main(['zenodo', '--path', str(path), '--sep-diagnostic', '--windows', '4'])
    output = capsys.readouterr().out
    assert 'Marginal sigma' in output and 'n/a' in output
    assert 'Chronological residual windows' in output
    assert path.read_text() == content


@pytest.mark.parametrize('arguments', [
    ['simulate', '--orbits', '-1'], ['simulate', '--points', '1'],
    ['compare', '--tolerance', 'nan'], ['simulate', '--system', 'PSR_B1913+16'],
    ['formula-cost', '--repeats', '0'], ['custom', '--path', 'missing', '--windows', '-1'],
    ['custom', '--path', 'missing', '--unit', 'days'], ['custom', '--path', 'missing'],
])
def test_cli_errors_are_actionable(arguments, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(arguments)
    assert exc.value.code == 2
    assert 'error:' in capsys.readouterr().err


def test_download_verifies_checksum_and_cleans_temporary_file(tmp_path, monkeypatch):
    content = b'example dataset bytes'
    monkeypatch.setattr(datasets, 'ZENODO_MD5', hashlib.md5(content).hexdigest())
    monkeypatch.setattr(datasets.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(content))
    target = tmp_path/'data'/'download.csv'
    assert datasets.ensure_zenodo_csv(target) == target
    assert target.read_bytes() == content
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize('failure', ['checksum', 'network', 'race'])
def test_failed_download_never_leaves_a_partial_or_overwrites_data(tmp_path, monkeypatch, failure):
    target = tmp_path/'data.csv'
    content = b'complete download'
    monkeypatch.setattr(datasets, 'ZENODO_MD5', 'bad' if failure == 'checksum' else hashlib.md5(content).hexdigest())
    def source(*args, **kwargs):
        if failure == 'network':
            raise URLError('unavailable')
        if failure == 'race':
            target.write_text('concurrent local data')
        return io.BytesIO(content)
    monkeypatch.setattr(datasets.urllib.request, 'urlopen', source)
    with pytest.raises((ValueError, OSError)):
        datasets.ensure_zenodo_csv(target)
    assert not list(tmp_path.glob('.zenodo-*'))
    if failure == 'race':
        assert target.read_text() == 'concurrent local data'
    else:
        assert not target.exists()


def test_cli_integration_failure_is_actionable(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise RuntimeError('Three-body integration failed: injected failure')
    monkeypatch.setattr(cli.RelationalThreeBodyIntegrator, 'integrate', fail)
    with pytest.raises(SystemExit) as exc:
        cli.main(['simulate'])
    assert exc.value.code == 2
    assert 'integration failed' in capsys.readouterr().err


def test_arecibo_keeps_data_inspection_separate_from_optional_simulation(tmp_path, monkeypatch, capsys):
    path=tmp_path/'arecibo.csv'
    path.write_text('# MJD (approx), residual, uncertainty (phase), j_GBT_1400, j_WSRT_1400\n'
                    '58000,0.001,0.001,0,0\n58002,-0.001,0.001,0,0\n')
    called=[]
    monkeypatch.setattr(cli,'cmd_simulate',lambda args: called.append(args))
    cli.main(['arecibo','--path',str(path)])
    assert not called
    cli.main(['arecibo','--path',str(path),'--simulate'])
    assert len(called)==1
    assert called[0].orbits == pytest.approx(2/cli.SYSTEMS['PSR_J0337+1715'].Pb1_days)
    output=capsys.readouterr().out
    assert 'no data-to-model fit' in output
    assert 'walk-forward' not in output.lower()


def test_former_nancay_command_rejects_false_observatory_attribution(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(['nancay'])
    assert exc.value.code == 2
    message=capsys.readouterr().err
    assert 'not Nancay' in message and 'arecibo' in message
