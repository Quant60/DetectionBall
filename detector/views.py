
import os
import shutil
from collections import deque
import cv2
import imageio.v2 as iio
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from ultralytics import YOLO
from .models import VideoDetection

import io
from django.http import FileResponse
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

SMOOTH_WINDOW = 5
SKIP_EVERY    = 2
GIF_WIDTH     = 1080

# Загружаем модель один раз при старте
MODEL = YOLO(str(settings.BASE_DIR / 'model' / 'best.pt'))

def index(request):
    if request.method == 'POST':
        vid = request.FILES.get('video')
        if not vid:
            return JsonResponse({'error': 'No file uploaded'}, status=400)

        # следующий порядковый номер
        idx = VideoDetection.objects.count() + 1

        # ─── 1) сохраняем входное видео в media/ ───
        ext   = os.path.splitext(vid.name)[1]
        in_fn = f"{idx}_input{ext}"
        in_fp = settings.MEDIA_ROOT / in_fn
        os.makedirs(in_fp.parent, exist_ok=True)
        with open(in_fp, 'wb') as f:
            for chunk in vid.chunks():
                f.write(chunk)

        # ── 1.1) и копируем то же видео в reports/videos/ ──
        rpt_v = settings.BASE_DIR / 'reports' / 'videos'
        os.makedirs(rpt_v, exist_ok=True)
        shutil.copy(str(in_fp), str(rpt_v / in_fn))

        # ─── 2) открываем видео и готовим параметры ───
        cap = cv2.VideoCapture(str(in_fp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        duration = (1.0 / fps) * SKIP_EVERY

        centers     = deque(maxlen=SMOOTH_WINDOW)
        sizes       = deque(maxlen=SMOOTH_WINDOW)
        last_center = None
        confidences = []
        frames_out  = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # пропускаем кадры
            if frame_idx % SKIP_EVERY != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            # детекция
            res = MODEL(frame, conf=0.25)[0]
            if res.boxes:
                b = res.boxes[0]
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0])
                confidences.append(conf)

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                side = max(x2 - x1, y2 - y1)
                centers.append((cx, cy))
                sizes.append(side)

                avg_cx   = int(sum(c[0] for c in centers) / len(centers))
                avg_cy   = int(sum(c[1] for c in centers) / len(centers))
                avg_side = int(sum(sizes)     / len(sizes))

                half = avg_side // 2
                sq_x1, sq_y1 = avg_cx-half, avg_cy-half
                sq_x2, sq_y2 = avg_cx+half, avg_cy+half
                cur_center = (avg_cx, avg_cy)

                # траектория белой линией
                if last_center:
                    cv2.line(frame, last_center, cur_center,
                             (255,255,255), 2, cv2.LINE_AA)
                last_center = cur_center

                # толстая зеленая рамка
                cv2.rectangle(frame,
                              (sq_x1, sq_y1), (sq_x2, sq_y2),
                              (0,255,0), 3, cv2.LINE_AA)

                # красный круг вокруг центра
                cv2.circle(frame, cur_center, int(avg_side/4),
                           (0,0,255), 2, cv2.LINE_AA)

                # текст с контуром
                text = f"Ball {conf:.2f}"
                font, scale, th = cv2.FONT_HERSHEY_DUPLEX, 0.9, 2
                (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
                tx1 = sq_x1
                ty1 = sq_y1 - tht - 8
                tx2 = tx1 + tw + 12
                ty2 = sq_y1

                overlay = frame.copy()
                cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2),
                              (0,255,0), cv2.FILLED)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

                cv2.putText(frame, text, (tx1+6, ty2-6),
                            font, scale, (0,0,0), th+2, cv2.LINE_AA)
                cv2.putText(frame, text, (tx1+6, ty2-6),
                            font, scale, (255,255,255), th, cv2.LINE_AA)

            # даунскейл для GIF
            h, w = frame.shape[:2]
            new_h = int(h * GIF_WIDTH / w)
            small = cv2.resize(frame, (GIF_WIDTH, new_h),
                               interpolation=cv2.INTER_AREA)
            frames_out.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        cap.release()

        # ─── 3) сохраняем GIF в media/ ───
        out_fn = f"{idx}_output.gif"
        out_fp = settings.MEDIA_ROOT / out_fn
        os.makedirs(out_fp.parent, exist_ok=True)
        iio.mimsave(
            out_fp,
            frames_out,
            duration=duration,
            subrectangles=True,
            quantizer='nq',
            optimize=True
        )

        # ─── 3.1) копируем GIF в reports/gifs/ ───
        rpt_g = settings.BASE_DIR / 'reports' / 'gifs'
        os.makedirs(rpt_g, exist_ok=True)
        shutil.copy(str(out_fp), str(rpt_g / out_fn))

        # ─── 4) сохраняем запись в БД ───
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        VideoDetection.objects.create(
            input_file=in_fn,
            output_file=out_fn,
            confidence=avg_conf
        )

        return JsonResponse({
            'output':     out_fn,
            'confidence': f"{avg_conf:.2f}"
        })

    return render(request, 'detector/index.html')

def download_report_pdf(request):
    data = [[
        'Timestamp',
        'Input File',
        'Output GIF',
        'Confidence'
    ]]
    for obj in VideoDetection.objects.order_by('-timestamp'):
        data.append([
            obj.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            obj.input_file,
            obj.output_file,
            f"{obj.confidence:.2f}"
        ])

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=20, rightMargin=20, topMargin=20, bottomMargin=20
    )

    styles = getSampleStyleSheet()
    elems = [
        Paragraph("VideoDetection History", styles['Title']),
        Spacer(1, 12)
    ]


    table = Table(
        data,
        repeatRows=1,
        colWidths=[100, 180, 180, 80],  # уменьшили 2 и 3
        hAlign='LEFT'
    )
    table.setStyle(TableStyle([
        ('BACKGROUND',      (0,0), (-1,0),      colors.HexColor('#74ABE2')),
        ('TEXTCOLOR',       (0,0), (-1,0),      colors.white),
        ('FONTNAME',        (0,0), (-1,0),      'Helvetica-Bold'),
        ('ALIGN',           (0,0), (-1,0),      'CENTER'),
        ('VALIGN',          (0,0), (-1,-1),     'MIDDLE'),
        ('FONTSIZE',        (0,0), (-1,-1),     10),
        ('GRID',            (0,0), (-1,-1),     0.5, colors.grey),
        ('ALIGN',           (0,1), (-1,-1),     'LEFT'),
        ('ROWBACKGROUNDS',  (0,1), (-1,-1),     [colors.whitesmoke, colors.lightgrey]),
    ]))
    elems.append(table)

    doc.build(elems)
    buffer.seek(0)
    return FileResponse(
        buffer,
        as_attachment=True,
        filename="VideoDetection_history.pdf"
    )
