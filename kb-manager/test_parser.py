import sys
sys.path.insert(0, r"C:\Users\10225\Downloads\KB\kb-manager")

from kb_manager.parsers import get_parser

parser = get_parser(r"C:\Users\10225\Downloads\KB\extracted\اشکالات سامانه اعتباریتو\EtebaritoProblems.xlsx")
if parser:
    doc = parser.parse(r"C:\Users\10225\Downloads\KB\extracted\اشکالات سامانه اعتباریتو\EtebaritoProblems.xlsx")
    with open("parser_test.txt", "w", encoding="utf-8") as f:
        f.write(f"Title: {doc.title}\n")
        f.write(f"Content preview: {doc.content[:500]}\n")
        if doc.sheets:
            for sheet in doc.sheets:
                f.write(f"Sheet: {sheet['name']}\n")
                f.write(f"Headers: {sheet['headers']}\n")
                for row in sheet['rows'][:2]:
                    f.write(f"Row: {row}\n")
    print("Done")
else:
    print("No parser found")