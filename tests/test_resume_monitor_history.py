import json
import os
import time

from sdfmpneo.training.monitor import build_resume_history, find_resume_sessions, read_jsonl


def write_rows(folder, rows, *, history=()):
    folder.mkdir(parents=True)
    with (folder/'metrics.jsonl').open('w', encoding='utf-8') as output:
        for row in rows:
            output.write(json.dumps(row)+'\n')
        output.write('{bad tail\n')
    (folder/'worker.settings.json').write_text(json.dumps({
        'history_sessions': [str(path) for path in history]
    }), encoding='utf-8')
    (folder/'worker.log').write_text(f'log:{folder.name}\n', encoding='utf-8')


def test_read_jsonl_ignores_bad_tail(tmp_path):
    path = tmp_path/'metrics.jsonl'
    path.write_text('{"revision": 1}\n{bad\n{"revision": 2}\n', encoding='utf-8')
    assert [row['revision'] for row in read_jsonl(path)] == [1, 2]


def test_resume_history_recovers_old_session_and_makes_counters_continuous(tmp_path):
    root = tmp_path/'case'
    logs = root/'logs'
    checkpoint = root/'results'/'model.stopped.npz'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'npz')

    first = logs/'20260908_010000_a'
    write_rows(first, [
        {'revision': 0, 'elapsed_s': 0.0, 'collocation_epoch': 0, 'mse': None},
        {'revision': 1, 'elapsed_s': 4.0, 'collocation_epoch': 0, 'mse': 4.0},
        {'revision': 2, 'elapsed_s': 10.0, 'collocation_epoch': 1, 'mse': 2.0,
         'state': 'stopped', 'checkpoint': str(checkpoint)},
    ])

    history = build_resume_history(logs, 'results/model.stopped.npz', root=root)
    assert history['sessions'] == [first]
    assert history['revision_offset'] == 2
    assert history['elapsed_offset_s'] == 10.0
    assert history['collocation_offset'] == 1
    assert [row['revision'] for row in history['rows'] if row.get('mse') is not None] == [1, 2]


def test_repeated_resume_uses_saved_lineage_and_newest_matching_session(tmp_path):
    root = tmp_path/'case'
    logs = root/'logs'
    checkpoint = root/'results'/'model.stopped.npz'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'npz')

    first = logs/'first'
    write_rows(first, [
        {'revision': 1, 'elapsed_s': 3.0, 'collocation_epoch': 0, 'mse': 3.0},
        {'revision': 2, 'elapsed_s': 8.0, 'collocation_epoch': 1, 'mse': 2.0,
         'state': 'stopped', 'checkpoint': str(checkpoint)},
    ])
    time.sleep(.01)
    second = logs/'second'
    write_rows(second, [
        {'revision': 1, 'elapsed_s': 2.0, 'collocation_epoch': 0, 'mse': 1.5},
        {'revision': 3, 'elapsed_s': 5.0, 'collocation_epoch': 2, 'mse': 1.0,
         'state': 'stopped', 'checkpoint': str(checkpoint)},
    ], history=[first])
    now = time.time()
    os.utime(first, (now-10, now-10))
    os.utime(second, (now, now))

    assert find_resume_sessions(logs, checkpoint, root=root) == [first, second]
    history = build_resume_history(logs, checkpoint, root=root)
    assert history['revision_offset'] == 5
    assert history['elapsed_offset_s'] == 13.0
    assert history['collocation_offset'] == 3
    plotted = [row['revision'] for row in history['rows'] if row.get('mse') is not None]
    assert plotted == [1, 2, 3, 5]
    assert history['rows'][-1]['elapsed_s'] == 13.0
    assert history['rows'][-1]['collocation_epoch'] == 3
