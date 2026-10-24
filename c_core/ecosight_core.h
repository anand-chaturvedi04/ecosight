/**
 * EcoSight AI - Native C Core Inference, Post-Processing, & Actuation Engine
 * Designed for low-latency (<5ms) deterministic edge deployment on smart bins.
 */
#ifndef ECOSIGHT_CORE_H
#define ECOSIGHT_CORE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_DETECTIONS 64
#define INPUT_SIZE 640
#define NUM_ANCHORS 8400

typedef enum {
    STREAM_ORGANIC = 0,
    STREAM_RECYCLABLE = 1,
    STREAM_GENERAL = 2,
    STREAM_NONE = 3
} WasteStream;

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    int class_id;
    char class_name[32];
    WasteStream stream;
    char stream_name[24];
} Detection;

typedef struct {
    int count;
    Detection detections[MAX_DETECTIONS];
    WasteStream primary_stream;
    char primary_stream_name[24];
    float top_confidence;
    int is_uncertain; // 1 if < 0.60
    
    // Cyber-Physical Actuator Telemetry
    uint8_t uart_packet[8];
    int packet_len;
    int servo_angle;

    // Sub-millisecond Execution Profile
    double preprocess_ms;
    double postprocess_ms;
    double total_ms;
    double fps;
} SegregationResult;

// Core Engine API
void ecosight_init(void);

void ecosight_preprocess_rgb(const uint8_t* src_rgb, int src_w, int src_h,
                            float* dst_chw, float* scale, int* pad_x, int* pad_y);

void ecosight_postprocess(const float* raw_tensor, int num_classes, int num_anchors,
                         float scale, int pad_x, int pad_y,
                         float conf_thresh, float iou_thresh,
                         SegregationResult* result);

float ecosight_compute_iou(const Detection* a, const Detection* b);

void ecosight_fast_nms(Detection* candidates, int count, float iou_thresh,
                       Detection* output, int* out_count);

WasteStream ecosight_classify_coco_stream(int class_id, char* name_buf, size_t buf_sz);

void ecosight_build_actuator_packet(WasteStream stream, int angle, uint8_t* out_pkt, int* out_len);

#ifdef __cplusplus
}
#endif

#endif // ECOSIGHT_CORE_H
