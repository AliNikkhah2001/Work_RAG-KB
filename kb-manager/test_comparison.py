from fastapi.testclient import TestClient
from kb_manager.web.app import app
import re

client = TestClient(app)
resp = client.get('/benchmarks/comparison')

# Check status
print('Status:', resp.status_code)

# Extract stat values
stats = re.findall(r'id="stat-(\w+)"[^>]*>([^<]+)', resp.text)
print('Stats found:', stats)

# Check version buttons
btns = re.findall(r'class="version-btn"[^>]*data-version="(\w+)"', resp.text)
print('Version buttons:', btns)

# Check if Chart.js is loaded
print('Chart.js in head:', 'chart.umd.min.js' in resp.text)

# Check script functions
print('initCharts in script:', 'initCharts' in resp.text)
print('selectVersion in script:', 'selectVersion' in resp.text)

# Check if versionData is embedded
print('versionData in script:', 'versionData' in resp.text)

# Check canvas elements
canvas = re.findall(r'<canvas id="(\w+)"', resp.text)
print('Canvas elements:', canvas)