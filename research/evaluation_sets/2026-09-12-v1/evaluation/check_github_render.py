"""Check GitHub's Markdown renderer preserves every table and bold rate."""
import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess

class Parsed(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = 0
        self.details = 0
        self.strong = None
        self.bold_rates = []
        self.heading = None
        self.headings = []
    def handle_starttag(self, tag, attrs):
        if tag == 'table': self.tables += 1
        if tag == 'details': self.details += 1
        if tag == 'strong': self.strong = ''
        if tag == 'h2': self.heading = ''
    def handle_data(self, data):
        if self.strong is not None: self.strong += data
        if self.heading is not None: self.heading += data
    def handle_endtag(self, tag):
        if tag == 'strong' and self.strong is not None:
            if re.fullmatch(r'[0-9.]+%', self.strong): self.bold_rates.append(self.strong)
            self.strong = None
        if tag == 'h2' and self.heading is not None:
            self.headings.append(self.heading.strip())
            self.heading = None

parser = argparse.ArgumentParser()
parser.add_argument('--markdown', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
body = args.markdown.read_text(encoding='utf8')
payload = json.dumps({'text': body, 'mode': 'gfm', 'context': 'LZhangGJ/Kaggriculture'}, ensure_ascii=False).encode()
result = subprocess.run(['gh', 'api', 'markdown', '--method', 'POST', '--input', '-'], input=payload, capture_output=True, timeout=60)
if result.returncode:
    print(result.stderr.decode('utf8', errors='replace')[:3000])
    print(result.stdout.decode('utf8', errors='replace')[:3000])
    raise SystemExit(result.returncode)
html = result.stdout.decode('utf8')
parsed = Parsed()
parsed.feed(html)
expected_tables = len(re.findall(r'^\|---', body, re.M))
expected_details = body.count('<details>')
expected_bold = Counter(re.findall(r'\*\*([0-9.]+%)\*\*', body))
expected_headings = re.findall(r'^## (.+)$', body, re.M)
assert parsed.tables == expected_tables, (parsed.tables, expected_tables)
assert parsed.details == expected_details, (parsed.details, expected_details)
assert Counter(parsed.bold_rates) == expected_bold
assert parsed.headings == expected_headings, (parsed.headings, expected_headings)
assert 'Retired holdout receipt' in html
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.with_suffix('.html').write_bytes(result.stdout)
receipt = dict(status='PASS', renderer='GitHub Markdown API, GFM',
    markdown_sha256=hashlib.sha256(args.markdown.read_bytes()).hexdigest(),
    rendered_html_sha256=hashlib.sha256(result.stdout).hexdigest(), markdown_bytes=args.markdown.stat().st_size,
    tables=parsed.tables, expandable_sections=parsed.details, highlighted_rate_cells=len(parsed.bold_rates), headings=parsed.headings)
args.out.write_bytes((json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode())
print(json.dumps(receipt))
