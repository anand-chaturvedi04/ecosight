/**
 * EcoSight AI - High-Performance Native C Core Engine
 */
#include "ecosight_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char* COCO_NAMES[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

void ecosight_init(void) {
    // Static state initialization
}

WasteStream ecosight_classify_coco_stream(int class_id, char* name_buf, size_t buf_sz) {
    if (class_id < 0 || class_id >= 80) {
        if (name_buf) snprintf(name_buf, buf_sz, "unknown");
        return STREAM_GENERAL;
    }

    const char* name = COCO_NAMES[class_id];
    if (name_buf) {
        snprintf(name_buf, buf_sz, "%s", name);
    }

    // Organics & Composts (Food items)
    if (class_id >= 46 && class_id <= 55) { // banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
        return STREAM_ORGANIC;
    }

    // Recyclables (Bottles, cans, glass, utensils, paper)
    if (class_id == 39 || class_id == 40 || class_id == 41 || // bottle, wine glass, cup
        class_id == 42 || class_id == 43 || class_id == 44 || // fork, knife, spoon
        class_id == 73) {                                      // book/paper
        return STREAM_RECYCLABLE;
    }

    // General Waste / E-waste
    return STREAM_GENERAL;
}

void ecosight_preprocess_rgb(const uint8_t* src_rgb, int src_w, int src_h,
                            float* dst_chw, float* scale, int* pad_x, int* pad_y) {
    float r_w = (float)INPUT_SIZE / (float)src_w;
    float r_h = (float)INPUT_SIZE / (float)src_h;
    float s = (r_w < r_h) ? r_w : r_h;
    *scale = s;

    int new_w = (int)roundf(src_w * s);
    int new_h = (int)roundf(src_h * s);
    *pad_x = (INPUT_SIZE - new_w) / 2;
    *pad_y = (INPUT_SIZE - new_h) / 2;

    int channel_stride = INPUT_SIZE * INPUT_SIZE;

    // Fill background with grey (114 / 255.0 = 0.447f)
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < channel_stride; i++) {
            dst_chw[c * channel_stride + i] = 0.447f;
        }
    }

    // Bilinear letterbox resize directly into CHW normalized floats
    for (int dy = 0; dy < new_h; dy++) {
        float sy = dy / s;
        int y_low = (int)sy;
        int y_high = (y_low + 1 < src_h) ? y_low + 1 : y_low;
        float y_diff = sy - y_low;

        int dst_row = (*pad_y + dy) * INPUT_SIZE + *pad_x;

        for (int dx = 0; dx < new_w; dx++) {
            float sx = dx / s;
            int x_low = (int)sx;
            int x_high = (x_low + 1 < src_w) ? x_low + 1 : x_low;
            float x_diff = sx - x_low;

            const uint8_t* p00 = src_rgb + (y_low * src_w + x_low) * 3;
            const uint8_t* p01 = src_rgb + (y_low * src_w + x_high) * 3;
            const uint8_t* p10 = src_rgb + (y_high * src_w + x_low) * 3;
            const uint8_t* p11 = src_rgb + (y_high * src_w + x_high) * 3;

            int dst_idx = dst_row + dx;

            for (int c = 0; c < 3; c++) {
                float top = (1.0f - x_diff) * p00[c] + x_diff * p01[c];
                float bot = (1.0f - x_diff) * p10[c] + x_diff * p11[c];
                float val = (1.0f - y_diff) * top + y_diff * bot;
                dst_chw[c * channel_stride + dst_idx] = val / 255.0f;
            }
        }
    }
}

float ecosight_compute_iou(const Detection* a, const Detection* b) {
    float x1 = (a->x1 > b->x1) ? a->x1 : b->x1;
    float y1 = (a->y1 > b->y1) ? a->y1 : b->y1;
    float x2 = (a->x2 < b->x2) ? a->x2 : b->x2;
    float y2 = (a->y2 < b->y2) ? a->y2 : b->y2;

    float inter_w = x2 - x1;
    float inter_h = y2 - y1;
    if (inter_w <= 0.0f || inter_h <= 0.0f) return 0.0f;

    float inter_area = inter_w * inter_h;
    float a_area = (a->x2 - a->x1) * (a->y2 - a->y1);
    float b_area = (b->x2 - b->x1) * (b->y2 - b->y1);
    float union_area = a_area + b_area - inter_area;

    return (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;
}

static int compare_detections(const void* a, const void* b) {
    float diff = ((const Detection*)b)->confidence - ((const Detection*)a)->confidence;
    return (diff > 0.0f) ? 1 : ((diff < 0.0f) ? -1 : 0);
}

void ecosight_fast_nms(Detection* candidates, int count, float iou_thresh,
                       Detection* output, int* out_count) {
    if (count <= 0) {
        *out_count = 0;
        return;
    }

    qsort(candidates, count, sizeof(Detection), compare_detections);

    uint8_t suppressed[512] = {0};
    int selected = 0;

    for (int i = 0; i < count && selected < MAX_DETECTIONS; i++) {
        if (suppressed[i]) continue;

        output[selected++] = candidates[i];

        for (int j = i + 1; j < count; j++) {
            if (!suppressed[j]) {
                if (candidates[i].class_id == candidates[j].class_id) {
                    if (ecosight_compute_iou(&candidates[i], &candidates[j]) > iou_thresh) {
                        suppressed[j] = 1;
                    }
                }
            }
        }
    }

    *out_count = selected;
}

void ecosight_build_actuator_packet(WasteStream stream, int angle, uint8_t* out_pkt, int* out_len) {
    out_pkt[0] = 0xAA; // Frame Header
    out_pkt[1] = 0x01; // Smart Bin Node ID
    out_pkt[2] = (uint8_t)stream; // Target Stream ID
    out_pkt[3] = (uint8_t)angle;  // Target Servo Angle (deg)
    out_pkt[4] = (uint8_t)(out_pkt[0] ^ out_pkt[1] ^ out_pkt[2] ^ out_pkt[3]); // XOR Checksum
    out_pkt[5] = 0x55; // Frame Footer
    *out_len = 6;
}

void ecosight_postprocess(const float* raw_tensor, int num_classes, int num_anchors,
                         float scale, int pad_x, int pad_y,
                         float conf_thresh, float iou_thresh,
                         SegregationResult* result) {
    memset(result, 0, sizeof(SegregationResult));

    Detection candidates[512];
    int cand_count = 0;

    // YOLOv8 raw output layout: [1, 4 + num_classes, 8400]
    // row 0: cx, row 1: cy, row 2: w, row 3: h, rows 4..4+C-1: class scores
    const float* cx_row = raw_tensor;
    const float* cy_row = raw_tensor + num_anchors;
    const float* w_row  = raw_tensor + 2 * num_anchors;
    const float* h_row  = raw_tensor + 3 * num_anchors;
    const float* cls_rows = raw_tensor + 4 * num_anchors;

    for (int a = 0; a < num_anchors; a++) {
        int best_cls = -1;
        float best_score = 0.0f;

        for (int c = 0; c < num_classes; c++) {
            float score = cls_rows[c * num_anchors + a];
            if (score > best_score) {
                best_score = score;
                best_cls = c;
            }
        }

        if (best_score >= conf_thresh && cand_count < 512) {
            float cx = cx_row[a];
            float cy = cy_row[a];
            float w  = w_row[a];
            float h  = h_row[a];

            float x1 = (cx - 0.5f * w - pad_x) / scale;
            float y1 = (cy - 0.5f * h - pad_y) / scale;
            float x2 = (cx + 0.5f * w - pad_x) / scale;
            float y2 = (cy + 0.5f * h - pad_y) / scale;

            Detection* d = &candidates[cand_count++];
            d->x1 = (x1 < 0.0f) ? 0.0f : x1;
            d->y1 = (y1 < 0.0f) ? 0.0f : y1;
            d->x2 = x2;
            d->y2 = y2;
            d->confidence = best_score;
            d->class_id = best_cls;
            d->stream = ecosight_classify_coco_stream(best_cls, d->class_name, sizeof(d->class_name));

            switch (d->stream) {
                case STREAM_ORGANIC: snprintf(d->stream_name, sizeof(d->stream_name), "Organic"); break;
                case STREAM_RECYCLABLE: snprintf(d->stream_name, sizeof(d->stream_name), "Recyclable"); break;
                default: snprintf(d->stream_name, sizeof(d->stream_name), "General"); break;
            }
        }
    }

    // Apply fast C NMS
    ecosight_fast_nms(candidates, cand_count, iou_thresh, result->detections, &result->count);

    // Aggregate winner
    if (result->count > 0) {
        float stream_scores[3] = {0.0f};
        float top_conf = 0.0f;

        for (int i = 0; i < result->count; i++) {
            WasteStream st = result->detections[i].stream;
            if (st <= STREAM_GENERAL) {
                stream_scores[st] += result->detections[i].confidence;
            }
            if (result->detections[i].confidence > top_conf) {
                top_conf = result->detections[i].confidence;
            }
        }

        WasteStream winner = STREAM_GENERAL;
        float max_s = stream_scores[STREAM_GENERAL];
        if (stream_scores[STREAM_RECYCLABLE] > max_s) {
            max_s = stream_scores[STREAM_RECYCLABLE];
            winner = STREAM_RECYCLABLE;
        }
        if (stream_scores[STREAM_ORGANIC] > max_s) {
            max_s = stream_scores[STREAM_ORGANIC];
            winner = STREAM_ORGANIC;
        }

        result->primary_stream = winner;
        result->top_confidence = top_conf;
        result->is_uncertain = (top_conf < 0.60f) ? 1 : 0;

        switch (winner) {
            case STREAM_ORGANIC:
                snprintf(result->primary_stream_name, sizeof(result->primary_stream_name), "Organic");
                result->servo_angle = 45;
                break;
            case STREAM_RECYCLABLE:
                snprintf(result->primary_stream_name, sizeof(result->primary_stream_name), "Recyclable");
                result->servo_angle = 90;
                break;
            default:
                snprintf(result->primary_stream_name, sizeof(result->primary_stream_name), "General");
                result->servo_angle = 135;
                break;
        }
    } else {
        result->primary_stream = STREAM_NONE;
        snprintf(result->primary_stream_name, sizeof(result->primary_stream_name), "None");
        result->top_confidence = 0.0f;
        result->is_uncertain = 0;
        result->servo_angle = 0;
    }

    ecosight_build_actuator_packet(result->primary_stream, result->servo_angle,
                                  result->uart_packet, &result->packet_len);
}
