"""Show beam expansion output for failing IVA queries."""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from kb_manager.query_expansion import SYNONYM_MAP_FINAL, generate_multi_queries

qs = [
    "چرا یکی از وام هایی که دارم قسطشون رو میدم، توی گزارش اعتباری من نیست؟",
    "چی کار کنم رتبم بهتر بشه؟",
    "گزارش من اشتباه داره. چی کار کنم؟",
    "من قسط وام خودم رو پرداخت کردم. اطلاعات من چه زمانی به روز میشه؟",
    "رتبه چه فرقی با امتیاز داره؟",
    "جزئیات قراردادهای منفی در گزارش یعنی چی؟",
]
for q in qs:
    print("\n#", q[:70])
    for i, beam in enumerate(generate_multi_queries(q, beam=5), 1):
        print(f"  {i}: {beam[:90]}")

print("\nmap size:", len(SYNONYM_MAP_FINAL))
for k in ["رو", "چی", "میشه", "چکم", "رتبم", "قسطشون", "توی", "دارم", "می‌دم", "من"]:
    print(f"  {k} ->", SYNONYM_MAP_FINAL.get(k))