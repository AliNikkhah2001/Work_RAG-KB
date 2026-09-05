import httpx, json, pathlib, time
out={}
s=time.time()
try:
    r=httpx.post("http://127.0.0.1:8000/search/api", json={"query":"\u0686\u06af\u0648\u0646\u0647 \u0627\u0645\u062a\u06cc\u0627\u0632 \u0627\u0639\u062a\u0628\u0627\u0631\u06cc \u0631\u0627 \u0628\u0628\u06cc\u0646\u0645\u061f","top_k":5}, timeout=600)
    out["status"]=r.status_code
    d=r.json(); out["n_results"]=len(d.get("final_results",[]))
    out["top"]=[{"chunk_id":x["chunk_id"][:8],"title":x["doc_title"],"score":x.get("rerank_score")} for x in d.get("final_results",[])[:3]]
except Exception as e:
    out["err"]=str(e)[:300]
out["sec"]=round(time.time()-s,1)
pathlib.Path("kb_warm.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
