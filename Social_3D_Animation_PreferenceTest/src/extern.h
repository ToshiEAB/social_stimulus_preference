#ifndef _EXTERN_H_
#define _EXTERN_H_

#include <string>
#include <vector>
#include <atomic>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

// Global flags for the communication between threads
extern std::atomic<bool> MaestroFlag;
extern std::atomic<bool> EndMaestroFlag;

// Global parameters
extern std::string FishID;
extern int ivSchedule;
extern int ivScheduleParameter;
extern int ivReversedSRLocation;
extern int ivSRprop;
extern int ivMaxSR;
//extern int ivSessionDuration;
extern int ivPhase1;
extern int ivPhase2;
extern int ivPhase3;
extern int ivPhase3_ReturnTime;
extern bool initPhase3Flag;
extern int phaseFlag;

extern int ivTargetCorner_1;
extern int ivTargetCorner_2;
extern int ivMotionSide;
extern std::string COM;
extern std::string SerialNumber;
extern const int SRduration;
extern bool SRarranged;

extern float BASE;
extern float SLOPE;
extern float INTERCEPT;
extern float UPPERLIMIT;

extern float YOLO_thresh;
extern float TF_thresh;

extern std::vector<int> sequence;
extern std::atomic<int> trial;
extern std::atomic<bool> idlingFlag; // Start with true
extern std::atomic<bool> automaticMotion;
extern std::atomic<bool> changeOpenGLPosition;
extern std::atomic<bool> resetOpenGLVideo;
extern std::atomic<float> OpenGLMotionSpeed;
extern cv::Mat videoMat;
extern cv::Mat openGLMat0;
extern cv::Mat openGLMat1;

//extern std::atomic<int> curCallback;

#endif
