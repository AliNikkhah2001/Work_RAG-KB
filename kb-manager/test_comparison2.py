from fastapi.testclient import TestClient
from kb_manager.web.app import app
import re

client = TestClient(app)
resp = client.get('/benchmarks/comparison')

# Check stats with better regex
stats = re.findall(r'id="stat-([^"]+)"[^>]*>([^<]+)', resp.text)
print('Stats found:', stats)

# Check version buttons - more flexible regex
btns = re.findall(r'data-version="([^"]+)"', resp.text)
print('Version buttons (data-version):', btns)

# Check the actual HTML around version buttons
version_section = re.search(r'Pipeline Version.*?Version', resp.text, re.DOTALL)
if version_section:
    print('Version section:', version_section.group()[:500])

# Check stat-hit-rate specifically
hit_rate = re.search(r'id="stat-hit-rate"', resp.text)
print('stat-hit-rate exists:', hit_rate is not None)

# Check all stat-value elements
stat_values = re.findall(r'class="stat-value"[^>]*id="([^"]+)"[^>]*>([^<]+)', resp.text)
print('All stat-values:', stat_values)