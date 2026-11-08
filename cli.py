#!/usr/bin/env python3
"""
CLI Tool for Quranic Audio Style & Reverb Cloner
(أداة سطر الأوامر لمحاكاة الهندسة الصوتية للتلاوات القرآنية)
"""
import argparse
import sys
import os
import shutil

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.dsp.pipeline import QuranicAudioPipeline


def main():
    parser = argparse.ArgumentParser(
        description="استوديو محاكاة الهندسة الصوتية للتلاوات القرآنية - سطر الأوامر (CLI)"
    )
    parser.add_argument("--ref", required=True, help="مسار ملف التلاوة المرجعية (Reference Audio)")
    parser.add_argument("--target", required=True, help="مسار ملف تلاوتك الخام (Target Dry Audio)")
    parser.add_argument("--out", default="output_master.wav", help="مسار حفظ التلاوة النهائية (WAV 24-bit)")
    parser.add_argument("--ir-out", default=None, help="مسار اختياري لحفظ ملف استجابة النبضة (extracted_ir.wav)")
    parser.add_argument("--eq-intensity", type=float, default=0.8, help="نسبة مطابقة الترددات (0.0 إلى 1.0)")
    parser.add_argument("--eq-engine", choices=["spectral", "matchering"], default="spectral", help="محرك مطابقة الترددات")
    parser.add_argument("--reverb-mode", choices=["parametric", "wiener"], default="parametric", help="خوارزمية استخلاص الصدى")
    parser.add_argument("--decay-scale", type=float, default=1.0, help="مقياس زمن تلاشي الصدى (0.5 إلى 2.0)")
    parser.add_argument("--reverb-mix", type=float, default=None, help="نسبة دمج الصدى (0.0 إلى 1.0 - افتراضياً تلقائي)")
    parser.add_argument("--no-ducking", action="store_true", help="تعطيل خفض الصدى التلقائي أثناء النطق")
    parser.add_argument("--no-clean", action="store_true", help="تعطيل فلتر التنقية المسبقة لصوت المستخدم")
    parser.add_argument("--sr", type=int, default=44100, help="معدل العينات (Sample Rate)")

    args = parser.parse_args()

    if not os.path.exists(args.ref):
        print(f"[خطأ] ملف التلاوة المرجعية غير موجود: {args.ref}")
        sys.exit(1)
    if not os.path.exists(args.target):
        print(f"[خطأ] ملف التلاوة الخام غير موجود: {args.target}")
        sys.exit(1)

    print("=" * 65)
    print(" استوديو محاكاة الهندسة الصوتية للتلاوات القرآنية - Quranic Audio Cloner")
    print("=" * 65)
    print(f"-> التلاوة المرجعية : {args.ref}")
    print(f"-> التلاوة الخام     : {args.target}")
    print(f"-> محرك الترددات    : {args.eq_engine} (شدة: {args.eq_intensity})")
    print(f"-> نمط الصدى         : {args.reverb_mode} (مقياس: {args.decay_scale}x)")
    print(f"-> خفض الصدى الذكي  : {'معطل' if args.no_ducking else 'مفعل'}")
    print("-" * 65)

    def on_progress(idx, name, pct):
        print(f"[{pct:5.1f}%] مرحلة {idx}: {name}")

    pipeline = QuranicAudioPipeline(sample_rate=args.sr)

    try:
        result = pipeline.process(
            ref_file_path=args.ref,
            target_file_path=args.target,
            clean_target=not args.no_clean,
            eq_intensity=args.eq_intensity,
            eq_engine=args.eq_engine,
            reverb_mode=args.reverb_mode,
            decay_scale=args.decay_scale,
            reverb_mix=args.reverb_mix,
            enable_ducking=not args.no_ducking,
            progress_callback=on_progress
        )

        # Copy generated master to destination
        shutil.copy2(result["files"]["final_master"], args.out)
        print(f"\n[نجاح] تم حفظ التلاوة المعالجة بنجاح في: {args.out}")

        if args.ir_out:
            shutil.copy2(result["files"]["extracted_ir"], args.ir_out)
            print(f"[نجاح] تم حفظ ملف البصمة الصوتية (IR) في: {args.ir_out}")

        print(f"-> زمن المعالجة: {result['elapsed_seconds']} ثانية")
        print(f"-> RT60 المقاس: {result['acoustic_analysis']['rt60_seconds']} ثانية")
        print(f"-> Pre-delay: {result['acoustic_analysis']['predelay_ms']} ms")
        print(f"-> Reverb Mix: {int(result['parameters']['reverb_mix'] * 100)}%")
        print("=" * 65)

    except Exception as e:
        print(f"\n[فشل] حدث خطأ أثناء المعالجة: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
