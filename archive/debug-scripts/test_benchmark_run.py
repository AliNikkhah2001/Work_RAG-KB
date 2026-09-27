from fastapi.testclient import TestClient
from kb_manager.web.app import app
import time, json
client=TestClient(app)
r=client.post("/benchmarks/run", data={"dataset":"test_questions.json","top_k":5,"sample_size":"5"})
print('post', r.status_code, r.headers.get('location'))
if r.status_code in [303,302]:
    loc=r.headers.get('location')
    print('loc', loc)
    # extract job id
    import re
    m=re.search(r'job=([a-f0-9]+)', loc)
    if m:
        job=m.group(1)
        print('job', job)
        for i in range(10):
            time.sleep(1.5)
            s=client.get(f"/benchmarks/status/{job}")
            print(f"poll {i}", s.json())
            if s.json().get('status')=='done':
                break
    else:
        print('no job')
else:
    print(r.text[:500])
