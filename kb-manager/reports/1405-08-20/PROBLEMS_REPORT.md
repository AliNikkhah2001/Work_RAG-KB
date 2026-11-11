# KB corpus 1405-08-20 — problems report

Generated during ingest-preflight. Supplementary material is listed for routing only; it must not be vectorized as general KB chunks.

## 1. Heading / schema mismatches

_None remaining after schemas v1.0.1._

## 2. Blank or corrupt QA rows (ignored on ingest; patch from CSV)

Count: **24**. Full list: [`blank_or_corrupt_qa_rows.csv`](blank_or_corrupt_qa_rows.csv).

Policy: these rows are **skipped completely** by the chunker (not written to the vector DB). Use the CSV as an editor patch list; after answers are filled in the source Excel, re-ingest those files.

- `پایگاه دانش/اشخاص حقیقی/اعتبارسنجی تسهیلات/بانک سوالات/بانک سوالات افراد در زمینه گزارش اعتبارسنجی تسهیلات-اصلی.xlsx`: 10
- `پایگاه دانش/اشخاص حقوقی/اعتبارسنجی تسهیلات/بانک سوالات افراد در زمینه گزارش اعتبارسنجی تسهیلات/بانک سوالات افراد در زمینه گزارش اعتبارسنجی تسهیلات.xlsx`: 9
- `پایگاه دانش/پایگاه دانش عمومی/سوالات عمومی.xlsx`: 3
- `پایگاه دانش/اشخاص حقیقی/گزارش اعتبارسنجی چک/بانک سوالات/بانک سوالات افراد در زمینه مباحث مختلف با گزارش اعتبارسنجی چک.xlsx`: 2

## 3. Routing / preprocess issues

| Item | Severity | Status / fix |
|---|---|---|
| Welcome `متن خوش آمدگویی.docx` was entity-routed and discarded | high | Fixed: text copied into `packs/canned/v1.0.2` greeting; ingest skips welcome docx |
| `واژگان معادل.xlsx` skipped without updating synonym map | high | Fixed: writes `data/synonym_map_glossary.json`, merged at query expansion; never vectorized |
| `سوالات نامربوط.xlsx` skipped (correct for vector DB) | medium | Summarized into guardrails prompts v1.0.1 for judge |
| `نکات قابل توجه چک.xlsx` headers `پاسخ مختصر/طولانی` missed crm_qa | high | Fixed in schemas v1.0.1 + detect_schema |
| Short article sheets (`SectionTitle`/`Section Title` only) missed articles | high | Fixed: articles `min_overlap: 2` + `require_normalized_any: [sectiontitle]` |
| `نردبان اعتباری.xlsx` raw header `سوال ` padded | low | Parser already strips; chunker now strips keys too |
| Taxonomy OR-token bleed (legal facilities → person; cheque → person) | high | Fixed: folder_tokens AND-match in taxonomy v1.0.1 |
| نردبان / کفیل had no graph node (files sit on domain folder) | medium | Fixed: append file stem to hierarchy + taxonomy nodes |
| Cross-folder QA dedup dropped person business panel after legal panel | medium | Fixed: dedup scoped by `folder_hierarchy` |
| Same-folder `جایگزین` bank fully duplicate of `اصلی` → 0 chunks | info | Keep: canonical bank wins; delete جایگزین from corpus if unused |
| Leading space in filename ` رسا- توضیحات...xlsx` | low | Rename in source drop when convenient |
| Welcome source typo «به شما کنم» | low | Corrected to «کمک کنم» in canned greeting |

## 4. Overlap & duplicate guidance

1. **Taxonomy overlap** — do not put bare `تسهیلات` alone in person tokens; require AND of `[حقیقی, تسهیلات]` (now coded). Prefer `filter_path` exact matches in retrieval.
2. **Same-folder duplicates** (`اصلی` vs `جایگزین`) — either keep one file only, or keep جایگزین only for answers that truly differ; corpus-wide question dedup inside the same folder path will drop identical questions.
3. **Cross-domain identical panels** (حقیقی/حقوقی پنل کسب‌وکار) — now kept separately via hierarchy-scoped dedup.
4. **Optional corpus hygiene** — move `نردبان اعتباری.xlsx` / `کفیل.xlsx` into named subfolders so the graph is visible in the zip without relying on stem injection.

## 5. Off-topic file (not vectorized) — category counts

- `دریافت گزارش و خدمات`: 10
- `حریم خصوصی و دسترسی`: 6
- `عوامل موثر بر امتیاز`: 6
- `امتیاز و رتبه اعتباری`: 4
- `تسهیلات، وامگیری و ضمانت`: 3
- `اصلاح و به روزرسانی اطلاعات`: 3
- `تسهیلات، وام‌گیری و ضمانت`: 2
- `اصلاح و بهروزرسانی اطلاعات`: 2
- `سوابق منفی و بدهیها`: 2

Sample rows:
- [دریافت گزارش و خدمات] Q: من در خصوص سامانه <فرابانک> <ملت> سوالی دارم  آیا شما می‌تونید کمکم کنید؟ / A: متاسفانه شرکت اعتبارسنجی فقط در مورد محصولات خود می‌تواند پاسخگوی سوالات شما باشد. لطفا برای مشکلات سامانه هر بانک، با پ
- [تسهیلات، وام‌گیری و ضمانت] Q: چه وثیقه ای لازمه که بتونم وام 300 تومنی <ملی> رو بگیرم؟ / A: متاسفانه شرکت اعتبارسنجی فقط در مورد محصولات خود می‌تواند پاسخگوی سوالات شما باشد. برای کسب اطلاعات از شرایط تسهیلات هر 
- [تسهیلات، وام‌گیری و ضمانت] Q: چنتا ضامن لازمه برا وام بانک <مهر اقتصاد>؟ / A: متاسفانه شرکت اعتبارسنجی فقط در مورد محصولات خود می‌تواند پاسخگوی سوالات شما باشد. برای کسب اطلاعات از شرایط تسهیلات هر 
- [حریم خصوصی و دسترسی] Q: کد ملی من <۱۲۳۴۵۶۷۸۹۰> هست، امتیازم چنده؟ / A: دریافت گزارش اعتباری مستلزم احراز هویت و اخذ رضایت صریح صاحب اطلاعات است. این چت‌بات امکان دریافت اطلاعات شخصی یا صدور گ
- [حریم خصوصی و دسترسی] Q: شماره موبایلم من <091223456789> هست، میشه بگید امتیازم چنده و برام گزارش بگیرید؟ / A: دریافت گزارش اعتباری مستلزم احراز هویت و اخذ رضایت صریح صاحب اطلاعات است. این چت‌بات امکان دریافت اطلاعات شخصی یا صدور گ
- [حریم خصوصی و دسترسی] Q: با این کد ملی برام گزارش اعتبارسنجی بگیر. / A: دریافت گزارش اعتباری مستلزم احراز هویت و اخذ رضایت صریح صاحب اطلاعات است. این چت‌بات امکان دریافت اطلاعات شخصی یا صدور گ
- [حریم خصوصی و دسترسی] Q: اسمم احمد احمدیه. گزارش اعتباری بگیر برام. / A: دریافت گزارش اعتباری مستلزم احراز هویت و اخذ رضایت صریح صاحب اطلاعات است. این چت‌بات امکان دریافت اطلاعات شخصی یا صدور گ
- [حریم خصوصی و دسترسی] Q: میشه با داشتن کد ملی همسرم یا دوستم گزارش اعتبارسنجیش رو برام بفرستید؟ / A: خیر، استعلام گزارش اعتباری هر شخص مستلزم احراز هویت و تأیید مستقیم خود فرد است. دریافت گزارش اعتباری سایر اشخاص بدون اجا
- [عوامل موثر بر امتیاز] Q: فرمول ریاضی دقیق محاسبه امتیاز اعتباری چیه؟ / A: جزئیات محاسباتی و ضرایب دقیق الگوریتم‌های هوش مصنوعی جزو پارامترهای داخلی و محرمانه مدل اعتبارسنجی است و امکان افشای آن 
- [عوامل موثر بر امتیاز] Q: وزن دقیق هر قسط در الگوریتم چقدره؟ / A: جزئیات محاسباتی و ضرایب دقیق الگوریتم‌های هوش مصنوعی جزو پارامترهای داخلی و محرمانه مدل اعتبارسنجی است و امکان افشای آن 
- [دریافت گزارش و خدمات] Q: قبوض آب و برق من نیومده، از کجا پیگیری کنم؟ / A: در حال حاضر بدهی قبوض خدماتی و عوارض شهرداری در گزارش اعتبارسنجی درج نمی‌شود و پیگیری و پرداخت آن‌ها باید از طریق سامانه
- [اصلاح و بهروزرسانی اطلاعات] Q: من چکم رو تسویه کردم، میشه همین الان سابقه چک برگشتی من رو از سیستم پاک کنید؟ / A: خیر، شرکت اعتبارسنجی ایران صرفاً دریافت‌کننده اطلاعات از مراجع رسمی است و امکان حذف یا تغییر دستی سوابق وجود ندارد. اطلا

## 6. Supplementary (deferred discussion)

Score tables, shareholders, data providers, loan titles, welcome, off-topic, glossary, limitations docx — handle after main KB fixes. Rule: supplementary must not be parsed into the general vector DB except where we explicitly promote a curated artifact (canned greeting, glossary synonym map, guardrails off-topic summary).
