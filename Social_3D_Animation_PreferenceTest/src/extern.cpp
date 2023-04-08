#include "extern.h"

std::atomic<bool> MaestroFlag;
std::atomic<bool> EndMaestroFlag;

// Global parameters
std::string FishID;
int ivSchedule;
int ivScheduleParameter;
int ivReversedSRLocation;
int ivSRprop;
int ivMaxSR;
//int ivSessionDuration;
int ivPhase1;
int ivPhase2;
int ivPhase3;
int ivPhase3_ReturnTime;
bool initPhase3Flag = true; // Start with true
int phaseFlag;

int ivTargetCorner_1;
int ivTargetCorner_2;
int ivMotionSide;
std::string COM;
std::string SerialNumber;
const int SRduration = 15000;
bool SRarranged = false;

float BASE;
float SLOPE;
float INTERCEPT;
float UPPERLIMIT;

float YOLO_thresh;
float TF_thresh;

std::vector<int> sequence;
std::atomic<int> trial(0);
std::atomic<bool> idlingFlag(true); // Start with true
std::atomic<bool> automaticMotion(false);
std::atomic<bool> changeOpenGLPosition(false);
std::atomic<bool> resetOpenGLVideo(false);
std::atomic<float> OpenGLMotionSpeed(0.f);
cv::Mat videoMat;
cv::Mat openGLMat0;
cv::Mat openGLMat1;

//std::atomic<int> curCallback(-1);