import os, glob, codecs, json

sys_out = codecs.open(r"C:\Users\10225\Downloads\KB\kb-manager\verify_all_unicode.txt", "w", "utf-8")

roots = {
    "31Tir1405": r"C:\Users\10225\Downloads\KB\extracted_new\31Tir1405",
    "1405-05-20": r"C:\Users\10225\Downloads\KB\1405-05-20",
}

from kb_manager.parsers.xlsx_parser import XlsxParser

engine = os.getenv("KB_XLSX_ENGINE", "calamine")
parser = XlsxParser(engine=engine)

total = 0
with_issues = 0
issue_examples = []

for label, root in roots.items():
    if not os.path.isdir(root):
        sys_out.write("SKIP %s (missing\n" % label)
        continue
    files = glob.glob(root + r"\**\*.xlsx", recursive=True)
    files = [f for f in files if not os.path.basename(f).startswith("~$")]
    sys_out.write("=== %s: %d files ===\n" % (label, len(files)))
    for fp in files:
        total += 1
        try:
            doc = parser.parse(fp)
            warnings = doc.metadata.get("integrity_warnings", [])
        except Exception as e:
            sys_out.write("  ERR %s: %s\n" % (os.path.basename(fp), e))
            with_issues += 1
            continue
        if warnings:
            with_issues += 1
            sys_out.write("  ISSUE %s: %s\n" % (os.path.basename(fp), warnings[0]))
            if len(issue_examples) < 10:
                issue_examples.append((os.path.basename(fp), warnings))

sys_out.write("\nTOTAL files: %d\n" % total)
sys_out.write("FILES WITH UNICODE ISSUES: %d\n" % with_issues)
sys_out.write("TOTAL CLEAN: %d\n" % (total - with_issues))
sys_out.close()
print("done")