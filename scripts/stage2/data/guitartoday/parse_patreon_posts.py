"""Parse GuitarToday Patreon batch JSONs into a CSV catalog.

Extracts: post title, Soundslice slice ID, YouTube video ID, and post URL
from the raw Patreon API responses.

Usage:
    python scripts/data/guitartoday/parse_patreon_posts.py
    python scripts/data/guitartoday/parse_patreon_posts.py --info   # summary only

Output: data/guitartoday/posts.csv
"""

import json
import csv
import re
import argparse
from pathlib import Path

from gtp import REPO_ROOT
BATCHES_DIR = REPO_ROOT / 'data' / 'guitartoday' / 'patreon_posts'
OUTPUT_CSV = REPO_ROOT / 'data' / 'guitartoday' / 'posts.csv'


def load_all_posts():
    posts = []
    for batch_file in sorted(BATCHES_DIR.glob('batch-*.json'), key=lambda f: int(f.stem.split('-')[1])):
        with open(batch_file) as f:
            data = json.load(f)
        posts.extend(data.get('data', []))
    return posts


def extract_soundslice_id(content):
    matches = re.findall(r'soundslice\.com/slices/([a-zA-Z0-9]+)/', content)
    return matches[0] if matches else ''


def extract_youtube_id(content, embed_url):
    combined = content + ' ' + (embed_url or '')
    matches = re.findall(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)', combined)
    return matches[0] if matches else ''


def parse_post(post):
    attrs = post.get('attributes', {})
    content = attrs.get('content_json_string', '') or ''
    embed = attrs.get('embed', {}) or {}

    return {
        'post_id': post.get('id', ''),
        'title': (attrs.get('title', '') or '').strip(),
        'published_at': (attrs.get('published_at', '') or '')[:10],
        'soundslice_id': extract_soundslice_id(content),
        'youtube_id': extract_youtube_id(content, embed.get('url', '')),
        'patreon_url': attrs.get('url', ''),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--info', action='store_true', help='Print summary only')
    args = parser.parse_args()

    posts = load_all_posts()
    print(f'Total posts: {len(posts)}')

    rows = [parse_post(p) for p in posts]

    # Deduplicate by soundslice_id (some posts appear in multiple batches)
    seen_ss = set()
    unique_rows = []
    for row in rows:
        if row['soundslice_id']:
            if row['soundslice_id'] in seen_ss:
                continue
            seen_ss.add(row['soundslice_id'])
        unique_rows.append(row)

    has_ss = sum(1 for r in unique_rows if r['soundslice_id'])
    has_yt = sum(1 for r in unique_rows if r['youtube_id'])
    has_both = sum(1 for r in unique_rows if r['soundslice_id'] and r['youtube_id'])
    has_neither = sum(1 for r in unique_rows if not r['soundslice_id'] and not r['youtube_id'])

    print(f'Unique entries: {len(unique_rows)}')
    print(f'  With Soundslice: {has_ss}')
    print(f'  With YouTube: {has_yt}')
    print(f'  With both: {has_both}')
    print(f'  With neither: {has_neither}')

    if args.info:
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ['post_id', 'title', 'published_at', 'soundslice_id', 'youtube_id', 'patreon_url']
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(unique_rows)

    print(f'\nCatalog written: {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
