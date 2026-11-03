"""Create local aggregate and failure-case reports without media."""
import json
from pathlib import Path
from modules.evaluation import summarize
root=Path(__file__).resolve().parent
source=root/"security_logs/trials.jsonl"
if not source.exists():
    print("No trials recorded yet. Use Test mode from the camera menu.")
else:
    records=[json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    report=summarize(records)
    destination=root/"security_logs/trial_report.json"
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    failures=[r for r in records if r["outcome"]=="inconclusive" or (r["kind"]=="live" and r["outcome"]=="rejected") or (r["kind"]!="live" and r["outcome"]=="accepted")]
    (root/"security_logs/failure_cases.json").write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# تقرير التجارب المحلية", "", "النتائج حسب التصنيف الذي أدخله المجرب. ليست شهادة أمان أو اختبارًا مستقلاً.", "", "| النوع / الظروف / المسار | العدد | قبول | رفض | غير محسوم | نسبة الخطأ في النتائج المحسومة |", "|---|---:|---:|---:|---:|---:|"]
    for group,c in report.items():
        rate="غير متاحة" if c["error_rate_decided"] is None else f'{100*c["error_rate_decided"]:.1f}%'
        safe=group.replace("|"," / ").replace("\n"," ")
        lines.append(f'| {safe} | {c["total"]} | {c["accepted"]} | {c["rejected"]} | {c["inconclusive"]} | {rate} |')
    lines.extend(["", "نسبة الخطأ لا تشمل الحالات غير المحسومة؛ اقرأ العددين معًا. تكرار التجربة على وجه واحد لا يثبت التعميم.", "", f"حالات الخطأ أو عدم الحسم المحفوظة نصيًا: {len(failures)}. انظر failure_cases.json.", "", "لا تُحفظ صور أو فيديو في هذه التقارير. الجلسات التي لم تبدأ لا تدخل الإحصاء."])
    (root/"security_logs/trial_report.md").write_text("\n".join(lines),encoding="utf-8")
    print("Reports saved in security_logs: trial_report.md, trial_report.json, failure_cases.json")
