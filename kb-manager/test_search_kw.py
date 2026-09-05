from kb_manager.web.routes.search import BM25, _tokenize
import sqlite3, json, pathlib
q='گزارش اعتباری'
print('tokens', _tokenize(q))
conn=sqlite3.connect('data/kb_test.db')
cur=conn.cursor()
cur.execute('SELECT id, content, keywords FROM chunks LIMIT 2')
for id, content, kw in cur.fetchall():
    try:
        kw_list=json.loads(kw) if kw else []
    except:
        kw_list=[]
    print('kw list', kw_list[:3])
    joined=' '.join(kw_list) if kw_list else ''
    print('joined', joined[:100])
    boosted=content+' '+joined+' '+joined+' '+joined
    print('boosted len', len(boosted), 'orig', len(content))
    break
print('keyword split test', 'a، b،c'.split('\u060c'))
print('expected strip?', [s.strip() for s in 'a، b،c'.split('\u060c')])
# check semantic.py keyword handling
with open('kb_manager/chunker/semantic.py', encoding='utf-8') as f:
    t=f.read()
    print('semantic has split', 'split(\"\\u060c\")' in t)
    # check if strip
    print('has strip?', '.strip()' in t[t.find('keywords=fields'):t.find('keywords=fields')+500])
# check benchmark_comparison vs reality
p=pathlib.Path('data/benchmark_comparison.json')
d=json.loads(p.read_text(encoding='utf-8'))
print('v4 overall', d['versions']['v4']['overall'])
print('v4 keyword_only', d['versions']['v4']['per_format']['keyword_only'])
