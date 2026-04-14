"""Fetch Soundslice notation data for all GuitarToday tabs.

Uses Playwright to load each slice page in a real browser, intercept the
data1.json and sync JSON network requests, and save them locally.

First run: opens a visible browser for you to log in to Soundslice via Patreon.
After login, saves session cookies for subsequent headless runs.

Usage:
    python scripts/fetch_guitartoday.py                    # first run: login
    python scripts/fetch_guitartoday.py --headless         # after login: headless
    python scripts/fetch_guitartoday.py --limit 5          # test with 5 slices
"""

import json
import os
import re
import time
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright

# scripts/data/guitartoday/ → 4 levels up to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BATCHES_DIR = REPO_ROOT / 'data' / 'guitartoday' / 'patreon-posts'
OUTPUT_DIR = REPO_ROOT / 'data' / 'guitartoday' / 'slices'
COOKIES_PATH = REPO_ROOT / 'data' / 'guitartoday' / 'soundslice_cookies.json'


def extract_slice_urls():
    """Extract unique Soundslice URLs from all batch JSON files."""
    urls = set()
    for batch_file in sorted(BATCHES_DIR.glob('batch-*.json')):
        text = batch_file.read_text()
        found = re.findall(r'https://www\.soundslice\.com/slices/([a-zA-Z0-9]+)/', text)
        urls.update(found)
    return sorted(urls)


def fetch_slice(context, slice_id, output_dir):
    """Open a fresh page, capture data1.json and sync JSON, then close."""
    slice_dir = output_dir / slice_id
    data_path = slice_dir / 'data.json'
    sync_path = slice_dir / 'sync.json'

    if data_path.exists():
        return 'skip'

    captured = {}

    def handle_response(response):
        url = response.url
        if 'files.soundslice.com/data/json/' in url:
            try:
                body = response.json()
                if 'data1.json' in url:
                    captured['data'] = body
                elif 'syncpoint' in url:
                    captured['sync'] = body
                else:
                    # Capture any other JSON from the data path
                    if 'data' not in captured and isinstance(body, dict) and 'bars' in body:
                        captured['data'] = body
                    elif 'sync' not in captured and isinstance(body, list):
                        captured['sync'] = body
            except Exception:
                pass

    # Create a new page with the listener attached BEFORE navigation
    page = context.new_page()
    page.on('response', handle_response)

    try:
        page.goto(f'https://www.soundslice.com/slices/{slice_id}/', timeout=30000)
        page.wait_for_timeout(5000)
    except Exception as e:
        page.close()
        return f'error: {e}'

    page.close()

    if 'data' not in captured:
        return 'no data captured'

    slice_dir.mkdir(parents=True, exist_ok=True)
    with open(data_path, 'w') as f:
        json.dump(captured['data'], f, indent=2)

    if 'sync' in captured:
        with open(sync_path, 'w') as f:
            json.dump(captured['sync'], f, indent=2)

    n_bars = len(captured['data'].get('bars', []))
    has_sync = 'sync' in captured
    return f'ok ({n_bars} bars, sync={"yes" if has_sync else "no"})'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', help='Run headless (after initial login)')
    parser.add_argument('--limit', type=int, default=None, help='Process only first N slices')
    args = parser.parse_args()

    slice_ids = extract_slice_urls()
    print(f'Found {len(slice_ids)} unique Soundslice slices')

    if args.limit:
        slice_ids = slice_ids[:args.limit]
        print(f'Limited to {args.limit}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context()

        # Load saved cookies if available (not needed for public slices, but just in case)
        if COOKIES_PATH.exists():
            cookies = json.loads(COOKIES_PATH.read_text())
            context.add_cookies(cookies)
            print('Loaded saved cookies')

        done = 0
        skipped = 0
        failed = 0

        for i, slice_id in enumerate(slice_ids):
            result = fetch_slice(context, slice_id, OUTPUT_DIR)

            if result == 'skip':
                skipped += 1
                status = 'SKIP'
            elif result.startswith('ok'):
                done += 1
                status = result
            else:
                failed += 1
                status = f'FAIL: {result}'

            print(f'[{i+1:3d}/{len(slice_ids)}] {slice_id}: {status}')

            if result != 'skip':
                time.sleep(1)

        browser.close()

    print(f'\n=== Summary ===')
    print(f'Downloaded: {done}')
    print(f'Skipped (already exists): {skipped}')
    print(f'Failed: {failed}')
    print(f'Output: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
