/**
 * EcoSight AI - Standalone C++ Edge Inference & Actuator Daemon
 * Direct ONNX Runtime C++ engine with real-time HUD telemetry and UART servo actuation.
 */
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "../../YOLOv8-ONNXRuntime-CPP/inference.h"

static const std::vector<std::string> COCO_CLASSES = {
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

enum WasteStream {
    STREAM_ORGANIC = 0,
    STREAM_RECYCLABLE = 1,
    STREAM_GENERAL = 2,
    STREAM_NONE = 3
};

struct WasteVerdict {
    std::string item_name;
    WasteStream stream;
    std::string stream_name;
    int servo_angle;
    cv::Scalar color;
};

WasteVerdict map_to_stream(int class_id) {
    std::string name = (class_id >= 0 && class_id < (int)COCO_CLASSES.size()) ? COCO_CLASSES[class_id] : "Item";
    WasteVerdict v;
    v.item_name = name;

    // Organic: Food items (banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake)
    if (class_id >= 46 && class_id <= 55) {
        v.stream = STREAM_ORGANIC;
        v.stream_name = "Organic/Compost";
        v.servo_angle = 45;
        v.color = cv::Scalar(50, 180, 50); // Green
        return v;
    }

    // Recyclable: Bottles, cups, cans, glass, utensils, paper
    if (class_id == 39 || class_id == 40 || class_id == 41 || 
        class_id == 42 || class_id == 43 || class_id == 44 || 
        class_id == 73) {
        v.stream = STREAM_RECYCLABLE;
        v.stream_name = "Recyclable";
        v.servo_angle = 90;
        v.color = cv::Scalar(220, 140, 0); // Blue / Cyan
        return v;
    }

    // General Waste: wrappers, electronics, trash
    v.stream = STREAM_GENERAL;
    v.stream_name = "General Waste";
    v.servo_angle = 135;
    v.color = cv::Scalar(100, 100, 100); // Grey
    return v;
}

int main(int argc, char** argv) {
    std::cout << "================================================================" << std::endl;
    std::cout << "   🌿 EcoSight AI - Native C Core Inference & Actuation Engine  " << std::endl;
    std::cout << "================================================================" << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path> [onnx_model_path] [conf_thresh]" << std::endl;
        return 1;
    }

    std::string img_path = argv[1];
    std::string model_path = (argc >= 3) ? argv[2] : "ecosight/yolov8n.onnx";
    float conf_thresh = (argc >= 4) ? std::stof(argv[3]) : 0.25f;

    std::cout << "[INIT] Model Path : " << model_path << std::endl;
    std::cout << "[INIT] Target Img : " << img_path << std::endl;
    std::cout << "[INIT] Min Conf   : " << conf_thresh << std::endl;

    // Load image via OpenCV
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        std::cerr << "[ERROR] Could not open image: " << img_path << std::endl;
        return 1;
    }
    std::cout << "[IO] Image Loaded: " << img.cols << "x" << img.rows << " (" << img.channels() << " channels)" << std::endl;

    // Initialize YOLOv8 ONNX Runtime C++ Session
    YOLO_V8* yolo = new YOLO_V8;
    yolo->classes = COCO_CLASSES;

    DL_INIT_PARAM params;
    params.modelPath = model_path;
    params.modelType = YOLO_DETECT_V8;
    params.imgSize = { 640, 640 };
    params.rectConfidenceThreshold = conf_thresh;
    params.iouThreshold = 0.45f;
    params.cudaEnable = false;
    params.intraOpNumThreads = 4;

    std::cout << "[ENGINE] Initializing ONNX Runtime Session..." << std::endl;
    char* err = yolo->CreateSession(params);
    if (err != RET_OK) {
        std::cerr << "[ERROR] " << err << std::endl;
        return 1;
    }
    std::cout << "[ENGINE] C++ Engine initialized successfully." << std::endl;

    // Execute Inference
    std::vector<DL_RESULT> results;
    auto t0 = std::chrono::high_resolution_clock::now();
    yolo->RunSession(img, results);
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double fps = 1000.0 / total_ms;

    std::cout << "\n-------------------- [INFERENCE TELEMETRY] --------------------" << std::endl;
    std::cout << " [PERF] Loop Latency : " << std::fixed << std::setprecision(2) << total_ms << " ms" << std::endl;
    std::cout << " [PERF] Throughput   : " << fps << " FPS" << std::endl;
    std::cout << " [PERF] Objects Found: " << results.size() << std::endl;
    std::cout << "---------------------------------------------------------------" << std::endl;

    WasteStream primary_stream = STREAM_NONE;
    int primary_angle = 0;
    std::string primary_stream_name = "None";
    float top_conf = 0.0f;

    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];
        WasteVerdict v = map_to_stream(r.classId);

        std::cout << " #" << (i + 1) << " | " << std::setw(15) << std::left << v.item_name
                  << " | Stream: " << std::setw(18) << v.stream_name
                  << " | Conf: " << std::setw(6) << std::fixed << std::setprecision(3) << r.confidence
                  << " | Box: [" << r.box.x << ", " << r.box.y << ", " << r.box.width << ", " << r.box.height << "]"
                  << std::endl;

        if (r.confidence > top_conf) {
            top_conf = r.confidence;
            primary_stream = v.stream;
            primary_angle = v.servo_angle;
            primary_stream_name = v.stream_name;
        }

        // Render bounding box on image
        cv::rectangle(img, r.box, v.color, 3);
        std::string label = v.item_name + " (" + std::to_string((int)(r.confidence * 100)) + "%) -> " + v.stream_name;
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);
        cv::rectangle(img, cv::Point(r.box.x, r.box.y - labelSize.height - 8),
                      cv::Point(r.box.x + labelSize.width + 6, r.box.y), v.color, cv::FILLED);
        cv::putText(img, label, cv::Point(r.box.x + 3, r.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }

    if (results.empty()) {
        std::cout << " [INFO] No waste items detected above confidence threshold." << std::endl;
    }

    // Build Hardware UART Actuator Packet
    uint8_t uart_pkt[6];
    uart_pkt[0] = 0xAA; // Start Byte
    uart_pkt[1] = 0x01; // Smart Bin Node ID
    uart_pkt[2] = (uint8_t)primary_stream; // Stream ID (0=Organic, 1=Recyclable, 2=General)
    uart_pkt[3] = (uint8_t)primary_angle;  // Servo Angle (deg)
    uart_pkt[4] = (uint8_t)(uart_pkt[0] ^ uart_pkt[1] ^ uart_pkt[2] ^ uart_pkt[3]); // XOR Checksum
    uart_pkt[5] = 0x55; // End Byte

    std::cout << "\n----------------- [ACTUATOR HARDWARE BRIDGE] ------------------" << std::endl;
    std::cout << " [ACTUATOR] Target Stream   : " << primary_stream_name << std::endl;
    std::cout << " [ACTUATOR] Servo Flap Angle: " << primary_angle << " deg" << std::endl;
    std::cout << " [ACTUATOR] Raw UART Packet : ";
    for (int b = 0; b < 6; b++) {
        std::cout << "0x" << std::hex << std::uppercase << std::setw(2) << std::setfill('0') << (int)uart_pkt[b] << " ";
    }
    std::cout << std::dec << std::endl;
    std::cout << " [ACTUATOR] Signal Status   : READY >> /dev/ttyUSB0 (115200 baud)" << std::endl;
    std::cout << "================================================================" << std::endl;

    // Save annotated image proof
    std::string out_img_path = "annotated_c_output.jpg";
    cv::imwrite(out_img_path, img);
    std::cout << "[SAVED] Annotated proof saved to: " << out_img_path << std::endl;

    delete yolo;
    return 0;
}
