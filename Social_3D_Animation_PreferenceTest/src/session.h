#ifndef _SESSION_H_
#define _SESSION_H_

// Standard libraries
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <queue>
#include <thread>
#include <atomic>
#include <experimental/filesystem>
#include <dirent.h>
#include <cmath>




// RealSense libraries
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>

// YOLO
#include "yolo_v2_class.hpp"

// TensorFlow
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#include "tensorflow/core/platform/init_main.h"

// Eigen (in Tensorflow folder)
#include "/home/tk/Documents/tensorflow/third_party/eigen3/Eigen/Core"
#include "/home/tk/Documents/tensorflow/third_party/eigen3/Eigen/LU"
#include "/home/tk/Documents/tensorflow/third_party/eigen3/Eigen/SVD"
#include "/home/tk/Documents/tensorflow/third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

// Custom
#include "extern.h"

#define show(X) std::cout << #X << "\n" << X << "\n\n";
#define adjOrig false

using namespace Eigen;

class SessionApp {
private:
    // Frame parameters
    const int COLOR_WIDTH = 640;
    const int COLOR_HEIGHT = 360;//480;
    const int COLOR_FPS = 60;
    const int DEPTH_WIDTH = 640;
    const int DEPTH_HEIGHT = 360;//480;
    const int DEPTH_FPS = 60;

    // Video frame parameters
    const int V_WIDTH = 1024;
    const int V_HEIGHT = 600;
    const float V_FPS = 1.f;

    // Aquarium size
    const int AQUARIUM_minX = -10;
    const int AQUARIUM_minY = -10;
    const int AQUARIUM_minZ = 0;
    const int AQUARIUM_X = 435;
    const int AQUARIUM_Y = 150;
    const int AQUARIUM_Z = 110;

    // Distortion correction
    const float CONSTANT = (1.0f - SLOPE) * BASE + INTERCEPT;

    // librealsense2
    rs2::pipeline pipeRS;
    rs2::config cfg;
    rs2::frameset frames;
    rs2::pipeline_profile profile;
    struct rs2_intrinsics intrin_color;
    struct rs2_intrinsics intrin_depth;
    struct rs2_extrinsics extrin_d2c;
    struct rs2_extrinsics extrin_c2d;

    rs2::frame color_frame;
    rs2::frame depth_frame;
    rs2::colorizer color_map;

    bool dev_flag = true;

    // OpenCV
    cv::Mat colorFrame;
    cv::Mat depthFrame;

    const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
    const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
    const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);
    const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
    const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);
    const cv::Scalar SCALAR_PINK = cv::Scalar(255.0, 102.0, 255.0);
    const cv::Scalar SCALAR_LIGHTGREEN = cv::Scalar(102.0, 255.0, 102.0);

    // Origins of coordinates
    cv::Point2f origin_px = {};
    cv::Point3f origin_pt = {}; // in millimeter
    bool MouseInEffect = false;

    // TensorFlow
    typedef std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_dict;
    std::unique_ptr<tensorflow::Session> session = NULL;
    MatrixXf scmap_head, scmap_body, scmap_tail;
    Tensor<float, 3, RowMajor> locref_head, locref_body, locref_tail;

    Eigen::array<long, 3> sHead = {0, 0, 0}, sBody = {0, 0, 1}, sTail = {0, 0, 2};
    Eigen::array<long, 3> lHead = {0, 0, 0}, lBody = {0, 0, 2}, lTail = {0, 0, 4}; // in steps of 2 give 2 values (x & y)

    // For minimizing errors
    float old_depth = 0.0f;
    bool prev_memoryflag = true;

    // Session Date & Time
    char DateTime_start[25] = {};
    char DateTime_end[25] = {};
    double Session_StartTime = 0.0;

    // Indepedent variables
    int ivSchedule_X = 0;
    int ivScheduleParameter_X = 0;
    int ivReversedSRLocation_X = 0;
    int ivPhase1_X = 0;
    int ivPhase2_X = 0;
    int ivPhase3_X = 0;
    int ivPhase3_ReturnTime_X = 0;


    int ivSRprop_X = 0;
    int ivMaxSR_X = 0;
    int ivTargetCorner_1_X = 0;
    int ivTargetCorner_2_X = 0;

    // Dependent variables
    int dvResponse[4] = {}; // 3 + 1
    int dvReinforcer = 0;
    int dvReinforcer_1 = 0;
    int dvReinforcer_2 = 0;
    int tmpResponse = 0;

    int realTime_now = 0;

    int lastResponse = 0;

    // Event Markers
    int NumOfEvents = 0;

    // XorShift RNG
    unsigned long seed[4] = {};

    // Fleshler-Hoffman distribution
    static const int n = 10;
    int Value_Variable = 0;
    int v = 0;
    int order = 0;
    int Iteration_Variable = 0;
    int rd[n + 1] = {};
    int vi[n + 1] = {};
    int Sumxser = 0;

    double Period_Start = 0.0;

    // QuasiRandom
    //std::vector<int> sequence; // Moved to extern
    //std::vector<int> sequence_video;
    int SRside = 0;

    int seqSR = 0;
    double FeederOnsetTime = 0.0;

    // OpenGL
    const int submonitor_X = 1280;
    const int submonitorRight_Y = 0;
    const int submonitorLeft_Y = 600;

    // Idling 3D animation
    int idlingOnset = 0; // Start with 0
    const int idlingCriterion = 100; // in msec

    // OpenGL video
    int OpenGLVideoOnset = 0;
    const int OpenGLVideoInterval = 30000; // 30 sec (Applies to only when ivMotionSide = 0-4

    // COD
    bool blnCOD = false; // False for Training with a single VI
    int CODpara = 1000;
    int COD = 0;
    int CODonset = 0;
    int lastResp = 0; // 0: None

    // struct
    struct Coordinates {
        int time;
        int headX;
        int headY;
        int z;
        int bodyX;
        int bodyY;
        int tailX;
        int tailY;
    };
    std::deque<Coordinates> deqCoordinates;

    struct DifferenceCriteria
    {
        int x;
        int y;
        int z;
    };
    struct DifferenceCriteria diffCr = { 25, 25, 25 };;

    struct CornerCriteria
    {
        int x_low;
        int x_high;
        int y_low;
        int y_high;
        int z_low;
        int z_high;
    };
    struct CornerCriteria cCr[4] = {}; // 3 + 1

    struct detection_data_t {
        cv::Mat cap_frame;
        std::shared_ptr<image_t> det_image;
        std::vector<bbox_t> result_vec;
        cv::Mat draw_frame;
        cv::Mat dpt_frame; // New for avoiding occasional null pointers in cvGetMat(); Unnecessary if not using cv::imshow("depthFrame", dpt_frame);
        bool new_detection;
        uint64_t frame_id;
        bool exit_flag;
        detection_data_t() : exit_flag(false), new_detection(false) {}
    };

    // Memory
    std::vector<std::string> EventMarkers;

public:
    SessionApp();
    ~SessionApp();
    void initRS();
    int run(int argc, char* argv[]);

private:
    void EndSession();
    int file_count_native(const std::string& dir);
    int SessionTime();
    void ScheduleCheck(int side);
    void EvaluateResponse(Coordinates& coordinates);
    void EventMarker(const char* EventType);
    void EventMarker(const char* EventType, const Coordinates& coordinates);
    int ValidateResponse_1(const Coordinates& coordinates);
    int ValidateResponse_2(const Coordinates& coordinates);
    int RealTime();
    void GetDateTime(char *DateTime);
    void saveOrigin();
    void AdjustOrigin();
    void checkOriginFile();
    static void mouseCallback(int event, int x, int y, int flags, void* userdata);
    void mouseCallback(int event, int x, int y, int flags);
    void WorldCoordinates(cv::Mat& mat_img, std::vector<cv::Point>& px, cv::Point* p_pt);
    void draw_boxes(cv::Mat& mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names, int current_det_fps, int current_cap_fps);
    void updateFrame();
    std::vector<std::string> objects_names_from_file(std::string const filename);
    void Fleshler_Hoffman(int Parameter);
    void QuasiRandom_video(std::vector<int>& seq, int max, int segment);
    void QuasiRandom(std::vector<int>& seq, int max, int ratio);
    void init_XorShift();
    unsigned long XorShift();
};

inline int Float2Int(const float& x) {
    return _mm_cvtss_si32(_mm_load_ss(&x));
}

inline int Double2Int(const double& x) {
    return _mm_cvtsd_si32(_mm_load_sd(&x));
}

inline double get_dtime() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)(ts.tv_sec) * 1000 + (double)(ts.tv_nsec) * 0.000001);
}

#endif
