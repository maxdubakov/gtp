"""Small shared helpers."""


def pct(num, den):
    return 100.0 * num / den if den else 0.0


def format_eta(seconds, suffix=''):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    eta = f'~{h}h{m:02d}m' if h > 0 else f'~{m}m'
    return f'{eta}{suffix}'
