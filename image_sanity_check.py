#!/usr/bin/env python3
"""Read-only image sanity check for the cleaned pothole YOLO dataset."""
from pathlib import Path
from collections import Counter
from statistics import median
import csv, math, sys

try:
    from PIL import Image, ImageFile
except ImportError:
    print('ERROR: Pillow is missing. Install with:')
    print('  "/home/CL502-13/Downloads/latestmergedproject2.v1i.yolo26/.venv/bin/python" -m pip install Pillow')
    sys.exit(1)

ImageFile.LOAD_TRUNCATED_IMAGES = False
PROJECT_ROOT = Path('/home/CL502-13/Downloads/latestmergedproject2.v1i.yolo26')
DATASET_ROOT = PROJECT_ROOT / 'final_cleaned_dataset_no_conflicts'
REPORT = PROJECT_ROOT / 'image_sanity_report.txt'
CSV_REPORT = PROJECT_ROOT / 'image_sanity_details.csv'
SPLITS = ('train', 'valid', 'test')
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

def percentile(values, p):
    if not values: return None
    vals = sorted(values); x = (len(vals)-1)*p; lo = math.floor(x); hi = math.ceil(x)
    return vals[lo] if lo == hi else vals[lo]*(hi-x) + vals[hi]*(x-lo)

def mib(n): return f'{n/(1024*1024):.2f} MiB' if n is not None else 'N/A'

def main():
    if not DATASET_ROOT.is_dir():
        sys.exit(f'ERROR: Dataset folder not found: {DATASET_ROOT}\nCheck the path before running again.')
    rows, errors = [], []
    widths, heights, pixels, sizes, ratios = [], [], [], [], []
    formats, modes = Counter(), Counter()
    summary = {}
    total_bytes = 0
    print('READ-ONLY IMAGE CHECK — dataset files will not be changed.')
    print(f'Dataset: {DATASET_ROOT}\n')
    for split in SPLITS:
        images_dir = DATASET_ROOT / split / 'images'
        labels_dir = DATASET_ROOT / split / 'labels'
        if not images_dir.is_dir() or not labels_dir.is_dir():
            errors.append(f'{split}: missing images or labels directory')
            continue
        paths = sorted(p for p in images_dir.rglob('*') if p.is_file() and p.suffix.lower() in EXTS)
        sw, sh, ss = [], [], []
        readable = empty = annotated = missing = bad = 0
        for idx, path in enumerate(paths, 1):
            rel = path.relative_to(DATASET_ROOT).as_posix()
            label = labels_dir / path.relative_to(images_dir).with_suffix('.txt')
            w = h = pix = ratio = size = None
            fmt = mode = 'UNKNOWN'
            flag = 'OK'
            try:
                size = path.stat().st_size
                with Image.open(path) as im: im.verify()
                with Image.open(path) as im:
                    im.load(); w, h = im.size; fmt = im.format or 'UNKNOWN'; mode = im.mode
                pix = w*h; ratio = w/h if h else None
                readable += 1; formats[fmt] += 1; modes[mode] += 1
                widths.append(w); heights.append(h); pixels.append(pix); sizes.append(size); ratios.append(ratio)
                sw.append(w); sh.append(h); ss.append(size); total_bytes += size
                if w < 320 or h < 320: flag = 'CHECK_SMALL_DIMENSION'
                elif w > 4096 or h > 4096: flag = 'CHECK_VERY_LARGE_DIMENSION'
            except Exception as e:
                bad += 1; flag = f'UNREADABLE: {type(e).__name__}: {e}'
                errors.append(f'{rel}: {flag}')
            if label.is_file():
                try:
                    content = label.read_text(encoding='utf-8').strip()
                    status = 'EMPTY_BACKGROUND' if not content else 'ANNOTATED'
                    if content: annotated += 1
                    else: empty += 1
                except Exception as e:
                    status = f'LABEL_READ_ERROR: {type(e).__name__}'
                    errors.append(f'{label}: {status}')
            else:
                status = 'MISSING'; missing += 1; errors.append(f'Missing label: {label}')
            rows.append({'split':split,'image':rel,'width_px':w,'height_px':h,'pixels':pix,
                         'aspect_ratio_w_over_h':round(ratio,4) if ratio is not None else '',
                         'format':fmt,'colour_mode':mode,'file_size_bytes':size,
                         'label_status':status,'image_check':flag})
            if idx % 2000 == 0: print(f'  {split}: inspected {idx:,}/{len(paths):,}')
        summary[split] = (len(paths), readable, bad, empty, annotated, missing, sw, sh, ss)
        print(f'{split.upper()}: images={len(paths):,}, readable={readable:,}, unreadable={bad:,}, '
              f'backgrounds={empty:,}, annotated={annotated:,}, missing labels={missing:,}')
    if rows:
        with CSV_REPORT.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    out = ['READ-ONLY IMAGE SANITY CHECK','='*68,f'Dataset: {DATASET_ROOT}',f'Images inspected: {len(rows):,}',f'Total image file size: {mib(total_bytes)}','', 'SPLIT SUMMARY','-'*68]
    for split in SPLITS:
        if split not in summary: out.append(f'{split.upper()}: NOT CHECKED (folder missing)'); continue
        n, rd, bad, bg, ann, miss, sw, sh, ss = summary[split]
        out.append(f'{split.upper()}: images={n:,}, readable={rd:,}, unreadable={bad:,}, backgrounds={bg:,}, annotated={ann:,}, missing labels={miss:,}')
        if sw: out.append(f'  width min/median/max={min(sw)}/{median(sw):.0f}/{max(sw)}; height min/median/max={min(sh)}/{median(sh):.0f}/{max(sh)}')
        if ss: out.append(f'  file size min/median/max={mib(min(ss))}/{mib(median(ss))}/{mib(max(ss))}')
    out += ['', 'OVERALL IMAGE CHARACTERISTICS','-'*68]
    if widths:
        out += [f'Width px: min={min(widths)}, p25={percentile(widths,.25):.0f}, median={median(widths):.0f}, p75={percentile(widths,.75):.0f}, max={max(widths)}',
                f'Height px: min={min(heights)}, p25={percentile(heights,.25):.0f}, median={median(heights):.0f}, p75={percentile(heights,.75):.0f}, max={max(heights)}',
                f'Pixels/image: min={min(pixels):,}, median={median(pixels):,.0f}, max={max(pixels):,}',
                f'Aspect ratio W/H: min={min(ratios):.3f}, median={median(ratios):.3f}, max={max(ratios):.3f}',
                f'File size: min={mib(min(sizes))}, median={mib(median(sizes))}, p95={mib(percentile(sizes,.95))}, max={mib(max(sizes))}']
    out += ['', 'IMAGE FORMATS','-'*68] + [f'{k}: {v:,}' for k,v in formats.most_common()]
    out += ['', 'COLOUR MODES','-'*68] + [f'{k}: {v:,}' for k,v in modes.most_common()]
    flagged = [r for r in rows if r['image_check'] != 'OK']
    out += ['', 'WARNINGS / ERRORS','-'*68, f'Small/very-large dimension flags: {len(flagged):,}']
    out += [f"{r['image_check']}: {r['image']}" for r in flagged[:100]]
    if len(flagged)>100: out.append(f'... {len(flagged)-100:,} more dimension flags listed in CSV')
    out.append(f'Errors (unreadable images, missing labels, etc.): {len(errors):,}')
    out += errors[:100]
    if len(errors)>100: out.append(f'... {len(errors)-100:,} more errors omitted from text report')
    out += ['', 'INTERPRETATION','-'*68,
            'This script only inspects files. It does not resize, convert, move, rename, or delete anything.',
            'Dimension flags are prompts for review, not automatic failures.',
            'Empty labels are counted as background images, but visual inspection is needed to confirm they contain no pothole.',
            'Metadata checks cannot confirm that bounding boxes correctly surround potholes.',
            f'Detailed CSV: {CSV_REPORT}']
    REPORT.write_text('\n'.join(out)+'\n', encoding='utf-8')
    print('\n'+'\n'.join(out))
    print(f'\nSaved report: {REPORT}\nSaved details CSV: {CSV_REPORT}\nDONE — dataset files were not modified.')

if __name__ == '__main__': main()
