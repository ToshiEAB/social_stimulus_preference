/*
Copyright: Toshikazu Kuroda at Huckle Co., Ltd./Aichi Bunkyo University (Currently at ATR as of 2023)


Setting up an appropriate computer environment is required for building this program.
It is assumed that you have the following OS and hardware installed on your computer although other options should work with minor tweaks.

OS:
    Ubuntu 18.04 LTS

Hardware:
    NVIDIA® Titan RTX GPU
    Intel® Depth Camera D435

Software (VERY picky about versions):
    CUDA 10.0
    cuDNN 7.4
    TensorrRT 5.1
    Bazel 0.19.2
    C++ library of Tensorflow GPU 1.13.1

Software (Less picky about versions):
    Librealsense SDK 2.29.0 (installed under ~/Documents. See CMakeLists.txt for details.)
    OpenCV 3.4.2
    CMake 3.15.1
    GCC 7.4.0 (but see below)

References:
    YOLO ver. 3: https://github.com/AlexeyAB/darknet
        - Model training is required using your own images
    DeepLabCut ver. 2.1: https://github.com/DeepLabCut/DeepLabCut
        - After training a model using your own images on their original Python program,
            1) Freeze the model to .pb file
            2) Edit the model so that it will have a sigmoid function at the end of the pipeline
            3) Convert the mode to TensorRT model for faster processing speed
    OpenGL-load model: https://github.com/WHKnightZ/OpenGL-Load-Model
    FindTensorFlow.cmake: https://github.com/PatWie/tensorflow-cmake/blob/master/cmake/modules/FindTensorFlow.cmake

Important notes:
    TensorFlow is compiled with GCC 4 whereas GCC 7 is installed on Utubtu so that YOLO will be compiled with GCC 7 by default.
    To use the same ABI, add -D_GLIBCXX_USE_CXX11_ABI=0 option when building TensorFlow Python, C, and C++ libraries with Bazel and also when building YOLO with Makefile

    A Python conda environment needs to be activated for runing cmake with CMakeLists.txt. See environment.yml for details.

*/

#include <iostream>
#include <string>
#include "extern.h"
#include "session.h"

const int NumOfFish = 9;
std::string listFishID[NumOfFish] = { "SensorGL", "646RWM", "647RWM" , "648RWM", "649RWM", "650RWF", "651RWF", "652RWF", "653RWF"};

int input() {
    int fishnumber = 0;
    int choice = 0;

    while (1) {
        // Select fish ID
        std::cout << "Select a fish ID\n";
        for (int i = 0; i < NumOfFish; i++) { std::cout << "   " << i << ": " << listFishID[i] << std::endl; };

        std::cout << "Number = ";
        for (; !((std::cin >> fishnumber) && (fishnumber >= 0) && (fishnumber < NumOfFish)); ) {
            std::cin.clear();
            std::cin.ignore();
            std::cout << "INVALID\n";
            std::cout << "Number = ";
        }

        std::cout << "\n";

        switch (fishnumber) {
            case 0:
                FishID = listFishID[fishnumber]; // "SensorGL"
                // ivMotionSide:
                //      -1 = Absent/Absent
                //      0 = Chasing-Right/Motionless-Left, 1 = Motionless-Right/Chasing-Left,
                //      2 = Motionless-Right/Motionless-Left
                //      3 = Motionless-Right/Absent-Left, 4 = Absent-Right/Motionless-Left
                //      5 = Chasing-Right/Absent-Left, 6 = Absent-Right/Chasing-Left
                //
                //      Don't use ivMotionSide = 0 to 6 in this program (Modify int SessionApp::run() and modelApp.h for using these options)
                //
                //      7 = Fleeing-Right/Chasing-Left, 8 = Chasing-Right/Fleeing-Left
                //      9 = Flipped-Right/Chasing-Left, 10 = Chasing-Right/Flipped-Left
                //      11 = Independent-Right/Chasing-Left, 12 = Chasing-Right/Independent-Left
                ivMotionSide = 11;
                OpenGLMotionSpeed = 6.5f;
                break;
            case 1:
                FishID = listFishID[fishnumber]; // "646RWM"
                ivMotionSide = -1;
                OpenGLMotionSpeed = 6.5f;
                break;
            case 2:
                FishID = listFishID[fishnumber]; // "647RWM"
                ivMotionSide = -1;
                OpenGLMotionSpeed = 6.5f;
                break;
            case 3:
                FishID = listFishID[fishnumber]; // "648RWM"
                ivMotionSide = -1;
                OpenGLMotionSpeed = 6.5f;
                break;
            case 4:
                FishID = listFishID[fishnumber]; // "649RWM"
                ivMotionSide = -1;
                OpenGLMotionSpeed = 6.5f;
                break;
            case 5:
                FishID = listFishID[fishnumber]; // "650RWF"
                ivMotionSide = -1;
                OpenGLMotionSpeed = 6.5f;
                break;
            case 6:
                FishID = listFishID[fishnumber]; // "651RWF"
                ivMotionSide = -1;
                OpenGLMotionSpeed = 6.5f;
                break;
            case 7:
                FishID = listFishID[fishnumber]; // "652RWF"
                ivMotionSide = -1;
                OpenGLMotionSpeed = 6.5f;
                break;
            case 8:
                FishID = listFishID[fishnumber]; // "653RWF"
                ivMotionSide = -1;
                OpenGLMotionSpeed = 6.5f;
                break;
        }
        // Irrelevant parameters in this program
        ivSchedule = 4; // 0: EXT, 1: VT, 2: FR, 3: VI, 4: Continuously on
        ivScheduleParameter = 30;
        ivSRprop = 40; // 0 -> all 1s (Left), 40 -> all 0s (Right)
        ivMaxSR = 40;
        ivReversedSRLocation = 0; // 0: Same location as response, 1: Reversed (Don't use these in programs with OpenGL)


        ivTargetCorner_1 = 1; // 1: Left compartment, 2: Center compartment, 3: Right compartment
        ivTargetCorner_2 = 3;

        ivPhase1 = 60000 * 10; // First 5 min for warm-up period
        ivPhase2 = ivPhase1 + 60000 * 5;
        ivPhase3 = ivPhase2 + 60000 * 5;
        ivPhase3_ReturnTime = ivPhase2 + 3000; // 3 sec (Irrelevant if using a blank/white background in Phases 3)

        YOLO_thresh = 0.3f;
        TF_thresh = 0.3f;

        COM = "/dev/ttyACM0"; // USB port number
        SerialNumber = "817612070806"; // Serial number of RealSense camera

        // Origin: Top-left corner on the upper layer
        BASE = 0.398f;
        SLOPE = 1.3831f;
        INTERCEPT = 0.0018555f;
        UPPERLIMIT = 0.50f;

        // Choice
        std::cout << "Select a choice\n";
        std::cout << "   0: Start\n";
        std::cout << "   1: Abort\n";
        std::cout << "   2: Return\n";
        std::cout << "Choice = ";
        for (; !((std::cin >> choice) && (choice >= 0) && (choice < 3)); ) {
            std::cin.clear();
            std::cin.ignore();
            std::cout << "INVALID\n";
            std::cout << "Choice = ";
        }

        std::cout << "\n";

        if (choice != 2) break;
    }
    return choice;
}

int main(int argc, char* argv[]) {
    std::cout << "Preference test for social stimulus\n";
    std::cout << "Copyright: Toshikazu Kuroda\n";
    std::cout << "\n";

    if (input()) return 0;

    SessionApp app;
    app.initRS();
    int* p_argc = &argc;
    char** p_argv = argv;
    app.run(*p_argc, p_argv);

    return 0;
}
