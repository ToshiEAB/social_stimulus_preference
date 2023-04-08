/*
 * Notes:
 * 1) YOLO should be included for minimizing false alarms (e.g., mirror-image of fish on glass). It also allows for multi-subjects detection.
 *
 *
 */

// Custom headers
#include "session.h"
#include "extern.h"
#include "maestro.h"
#include "modelApp.h"

measures extern_target0;
measures extern_target1;

void thread_opengl(int argc, char **argv) {
    modelApp app;
    app.run(argc, argv);
}

int SessionApp::run(int argc, char* argv[]) {

    if (!dev_flag) {
        std::cout << "Error: No appropriate camera is connected\n";
        return -1;
    }

    // Loading a TensorFlow model
    const std::string graph_fn = "../train/trt_graph.pb";

    tensorflow::port::InitMain(argv[0], &argc, &argv);
    tensorflow::GraphDef graph_def;
    TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def));

    // Set TensorFlow options and create a TensorFlow session
    auto options = tensorflow::SessionOptions();
    options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5); // Set % of GPU memory allocation
    options.config.mutable_gpu_options()->set_allow_growth(true); // Allocate only as much GPU memory based on runtime allocations
    session.reset(tensorflow::NewSession(options));
    TF_CHECK_OK(session->Create(graph_def));

    // Create a tensor and its pointer (The pointer method speeds up the conversion from tensorflow::Tensor to cv::Mat)
    tensorflow::TensorShape data_shape({1, COLOR_HEIGHT, COLOR_WIDTH, 3}); // Depth = 3
    tensorflow::Tensor im(tensorflow::DT_FLOAT, data_shape);
    im.flat<float>().setZero();
    float *p = im.flat<float>().data();

    // Loading a YOLO model
    std::string names_file = "../data/obj.names";
    std::string cfg_file = "../YOLO_models/yolov3_zebrafish.cfg";
    std::string weights_file = "../YOLO_models/yolov3_zebrafish_best.weights";

    // Create a YOLO detector
    Detector detector(cfg_file, weights_file);
    std::vector<std::string> obj_names = objects_names_from_file(names_file);

    // OpenCV
    cv::setUseOptimized(true);

    // Set a callback function for mouse click
    cv::namedWindow("colorFrame");
    cv::moveWindow("colorFrame", 400, 100);
    cv::setMouseCallback("colorFrame", &SessionApp::mouseCallback, this);

    // Set the location of OpenCV windows
    //cv::namedWindow("depthFrame");
    //cv::moveWindow("depthFrame", COLOR_WIDTH + 50, 100);


    // SDs
    cv::Mat blank(V_HEIGHT, V_WIDTH, CV_8UC3, cv::Scalar(255.0, 255.0, 255.0)); // White
    //cv::Mat blank_SR(V_HEIGHT, V_WIDTH, CV_8UC3, cv::Scalar(0.0, 0.0, 0.0)); // Black
    //cv::Mat blank_SR(V_HEIGHT, V_WIDTH, CV_8UC3, cv::Scalar(255.0, 255.0, 255.0)); // White

    cv::namedWindow("Left");
    cv::namedWindow("Right");
    cv::moveWindow("Left", submonitor_X, submonitorLeft_Y);
    cv::moveWindow("Right", submonitor_X, submonitorRight_Y);

    // Retrieve the 3D origin from the previous session
    checkOriginFile();
    std::cout << "\n" << "World (in mm): X = " << origin_pt.x << ", Y = " << origin_pt.y << ", Z = " << origin_pt.z << "\n\n";

    if (adjOrig) {
        while (1) {
            updateFrame();
            if (MouseInEffect) AdjustOrigin();
            if (colorFrame.rows != 0 && colorFrame.cols != 0) cv::imshow("colorFrame", colorFrame);
            //if (depthFrame.rows != 0 && depthFrame.cols != 0) cv::imshow("depthFrame", depthFrame);
            int c = cv::waitKey(1);
            if (c == 's') break;
            if (c == 27) return 0;
        }

        // Save the 3D origin
        saveOrigin();

    } else {
        while(1) {
            updateFrame();
            if (colorFrame.rows != 0 && colorFrame.cols != 0 && depthFrame.rows != 0 && depthFrame.cols != 0) break;
        }
    }

    // Thread for OpenGL
    std::thread t1(thread_opengl, argc, argv);

    std::cout << "\n" << "Session Start\n" << "Terminate by pressing the ESC key\n";

    // Date & Time
    GetDateTime(DateTime_start);
    Session_StartTime = get_dtime();

    // Fleshler & Hoffman (Not used in this program)
    Fleshler_Hoffman(ivScheduleParameter_X);
    int VItimer = rd[Value_Variable];
    Period_Start = get_dtime();

    // Time measurement
    std::chrono::steady_clock::time_point steady_start, steady_end;

    uint64_t frame_id = 0;
    int fps_cap_counter = 0;
    int fps_det_counter = 0;
    int current_fps_cap = 0;
    int current_fps_det = 0;
    bool exit_flag = false;

    detection_data_t detection_data;
    detection_data = detection_data_t();
    while(1) {
        try {
            while (1) {
                updateFrame();
                detection_data.cap_frame = colorFrame;
                detection_data.dpt_frame = depthFrame;

                fps_cap_counter++;
                detection_data.frame_id = frame_id++;

                if (detection_data.cap_frame.empty() || exit_flag) {
                    std::cout << "  exit_flag: true" << std::endl;
                    detection_data.exit_flag = true;
                    detection_data.cap_frame = cv::Mat(cv::Size(COLOR_WIDTH ,COLOR_HEIGHT), CV_8UC3);
                    //maestro.turn_off_HL();
                    EndMaestroFlag = true;
                    EventMarker("99");
                    EndSession();
                    break;
                }

                 std::shared_ptr<image_t> det_image;
                det_image = detector.mat_to_image_resize(detection_data.cap_frame);
                detection_data.det_image = det_image;

                std::vector<bbox_t> result_vec;
                if(det_image) result_vec = detector.detect_resized(*det_image, COLOR_WIDTH, COLOR_HEIGHT, YOLO_thresh, true);  // true
                fps_det_counter++;

                detection_data.new_detection = true;
                detection_data.result_vec = result_vec;

                cv::Mat draw_frame = detection_data.cap_frame;
                //int frame_story = std::max(5, current_fps_cap.load());
                int frame_story = std::max(5, current_fps_cap);
                result_vec = detector.tracking_id(result_vec, true, frame_story, 40);

                if (!draw_frame.empty()) {
                    /* TensorFlow (start) */
                    // Create a dummy Mat for converting cv::Mat (rgbFrame) to tensorflow::Tensor (im) using a pointer method
                    cv::Mat img(COLOR_HEIGHT, COLOR_WIDTH, CV_32FC3, p);
                    cv::Mat rgbFrame; // Moved from session.h
                    cv::cvtColor(draw_frame, rgbFrame, CV_BGR2RGB); // *** This conversion is VERY important to be consistent with Python program
                    rgbFrame.convertTo(img, CV_32FC3);

                    tensor_dict feed_dict = { {"fifo_queue_Dequeue", im},}; // fifo_queue_Dequeue is the entry point of TensorFlow model
                    std::vector<tensorflow::Tensor> outputs;
                    TF_CHECK_OK(session->Run(feed_dict, {"sigmoid", "pose/locref_pred/block4/BiasAdd"}, {}, &outputs)); // sigmoid correponds to confidence (probability) whereas BiasAdd corresponds to the detected location (x, y) on 2D image

                    int64 dimSize_0_1, dimSize_0_2, dimSize_0_3, dimSize_1_1, dimSize_1_2, dimSize_1_3;
                    dimSize_0_1 = outputs[0].dim_size(1);
                    dimSize_0_2 = outputs[0].dim_size(2);
                    dimSize_0_3 = outputs[0].dim_size(3);
                    dimSize_1_1 = outputs[1].dim_size(1);
                    dimSize_1_2 = outputs[1].dim_size(2);
                    dimSize_1_3 = outputs[1].dim_size(3);

                    TensorMap entireScmap = TensorMap<Tensor<float, 3, RowMajor>>(outputs[0].flat<float>().data(), dimSize_0_1, dimSize_0_2, dimSize_0_3);
                    TensorMap entireLocref = TensorMap<Tensor<float, 3, RowMajor>>(outputs[1].flat<float>().data(), dimSize_1_1, dimSize_1_2, dimSize_1_3);

                    // Make slices of entireScmap & entireLocref
                    Eigen::array<long, 3> exScm {dimSize_0_1 , dimSize_0_2, 1}; // 1 for confidence being single values
                    Tensor<float, 3, RowMajor> sc_head = entireScmap.slice(sHead, exScm);
                    Tensor<float, 3, RowMajor> sc_body = entireScmap.slice(sBody, exScm);
                    Tensor<float, 3, RowMajor> sc_tail = entireScmap.slice(sTail, exScm);
                    scmap_head = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(sc_head.data(), dimSize_0_1, dimSize_0_2);
                    scmap_body = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(sc_body.data(), dimSize_0_1, dimSize_0_2);
                    scmap_tail = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(sc_tail.data(), dimSize_0_1, dimSize_0_2);

                    Eigen::array<long, 3> exLoc {dimSize_1_1, dimSize_1_2, 2}; // in steps of 2 give 2 values (x & y)
                    locref_head = entireLocref.slice(lHead, exLoc);
                    locref_body = entireLocref.slice(lBody, exLoc);
                    locref_tail = entireLocref.slice(lTail, exLoc);

                    std::vector<tensorflow::Tensor>().swap(outputs);

                    /* TensorFlow (end) */
                }

                draw_boxes(draw_frame, result_vec, obj_names, current_fps_det, current_fps_cap);
                detection_data.result_vec = result_vec;
                detection_data.draw_frame = draw_frame;

                // Used to be the main loop
                steady_end = std::chrono::steady_clock::now();
                float time_sec = std::chrono::duration<double>(steady_end - steady_start).count();
                if (time_sec >= 1) {
                    //current_fps_det = fps_det_counter.load() / time_sec;
                    //current_fps_cap = fps_cap_counter.load() / time_sec;
                    current_fps_det = fps_det_counter / time_sec;
                    current_fps_cap = fps_cap_counter / time_sec;
                    steady_start = steady_end;
                    fps_det_counter = 0;
                    fps_cap_counter = 0;
                }

                if (!detection_data.draw_frame.empty()) {
                    cv::imshow("colorFrame", detection_data.draw_frame);
                }
//            if (!detection_data.dpt_frame.empty()) {
//                cv::imshow("depthFrame", detection_data.dpt_frame);
//            }

                // While idling
                if (idlingFlag) {
                    if (!automaticMotion) {
                        if (RealTime() - idlingOnset > idlingCriterion) {
                            automaticMotion = true;
                        }
                    }
                } else {
                    idlingFlag = true;
                    idlingOnset = RealTime();
                }

                realTime_now = RealTime();

                // Use imshow only in the main thread
                switch (phaseFlag) {
                    case 1:
                        if (realTime_now >= ivPhase1_X) phaseFlag = 2;
                            cv::imshow("Right", blank);
                            cv::imshow("Left", blank);
                        break;
                    case 2:
                        if (realTime_now >= ivPhase2_X) phaseFlag = 3;

                        switch (ivMotionSide) {
                            case -1: // No fish/No fish
                                cv::imshow("Right", blank);
                                cv::imshow("Left", blank);
                                break;
                            case 0: // Motion-Right/Motionless-Left
                                cv::imshow("Right", openGLMat0);
                                cv::imshow("Left", videoMat);
                                break;
                            case 1: // Motionless-Right/Motion-Left
                                cv::imshow("Right", videoMat);
                                cv::imshow("Left", openGLMat0);
                                break;
                            case 2: // Motionless-Right/Motionless-Left
                                cv::imshow("Right", videoMat);
                                cv::imshow("Left", videoMat);
                                break;
                            case 3: // Motionless-Right/No fish-Left
                                cv::imshow("Right", videoMat);
                                cv::imshow("Left", blank);
                                break;
                            case 4: // No fish-Right/Motionless-Left
                                cv::imshow("Right", blank);
                                cv::imshow("Left", videoMat);
                                break;
                            case 5: // Motion-Right/No fish-Left
                                cv::imshow("Right", openGLMat0);
                                cv::imshow("Left", blank);
                                break;
                            case 6: // No fish-Right/Motion-Left
                                cv::imshow("Right", blank);
                                cv::imshow("Left", openGLMat0);
                                break;
                            case 7: // Fleeing-Right/Chasing-Left,
                                cv::imshow("Right", openGLMat1); // motionType = 6 (Fleeing)
                                cv::imshow("Left", openGLMat0); // motionType = 0 (Chasing)
                                break;
                            case 8: // Chasing-Right/Fleeing-Left
                                cv::imshow("Right", openGLMat0);
                                cv::imshow("Left", openGLMat1);
                                break;
                            case 9: // Flipped-Right/Chasing-Left
                                cv::imshow("Right", openGLMat1);
                                cv::imshow("Left", openGLMat0);
                                break;
                            case 10: // Chasing-Right/Flipped-Left
                                cv::imshow("Right", openGLMat0);
                                cv::imshow("Left", openGLMat1);
                                break;
                            case 11: // Independent-Right/Chasing-Left
                                cv::imshow("Right", openGLMat1);
                                cv::imshow("Left", openGLMat0);
                                break;
                            case 12: // Chasing-Right/Independent-Left
                                cv::imshow("Right", openGLMat0);
                                cv::imshow("Left", openGLMat1);
                                break;
                        }
                        break;
                    case 3:
                        cv::imshow("Right", blank);
                        cv::imshow("Left", blank);

                        if (realTime_now >= ivPhase3_X) exit_flag = true;
                        break;
                }

                switch (ivMotionSide) { // Applies only to Motionless (For avoiding bugs when ivMotionSide >= 7)
                    case 11:
                    case 12:
                        // Do nothing
                        break;
                    default:
                        if (realTime_now - OpenGLVideoOnset >= OpenGLVideoInterval) {
                            OpenGLVideoOnset = RealTime();
                            resetOpenGLVideo = true;
                        }
                        break;
                }


                int key = cv::waitKey(V_FPS);
                //if (key == 'p') while (true) if (cv::waitKey(100) == 'p') break; // Pause
                if (key == 27) { exit_flag = true;}
            }
            if (t1.joinable()) t1.join();
            std::cout << "OpenGL ended\n";

            break;
        }
        catch (std::exception &e) { std::cerr << "exception: " << e.what() << "\n"; getchar(); }
        catch (...) { std::cerr << "unknown exception \n"; getchar(); }
    }


    // Close TensorFlow session & Release memory
    session->Close();
    session.reset();

    // Close RealSense
    pipeRS.stop();

    return 0;
}

void SessionApp::draw_boxes(cv::Mat& mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names, int current_det_fps = -1, int current_cap_fps = -1) {

    for (auto &i : result_vec) {

        /* TensorFlow (start) */
        unsigned int x, y, w, h;
        x = i.x/8; // Divided by 8 because scmap is 8 times smaller than original frame
        y = i.y/8;
        w = i.w/8;
        h = i.h/8;

        MatrixXf::Index maxloc_row, maxloc_col;
        MatrixXf maxloc = MatrixXf::Zero(1, 2);
        MatrixXf offset = MatrixXf::Zero(1, 2);
        MatrixXf cst = MatrixXf::Constant(1, 2, 0.5f);
        MatrixXi pose_int = MatrixXi::Zero(1, 2);

        cv::Point pt[3] = {};
        cv::Point* p_pt = pt;

        MatrixXf s_head = scmap_head.block(y, x, h, w);

        if (s_head.maxCoeff() > TF_thresh) {
            s_head.maxCoeff(&maxloc_row, &maxloc_col);
            maxloc << maxloc_col + x, maxloc_row + y;
            offset << locref_head(maxloc(1), maxloc(0), 0), locref_head(maxloc(1), maxloc(0), 1);
            offset *= 7.2801f;
            MatrixXf pose = (maxloc + cst) * 8.0f + offset;
            pose_int = pose.cast<int>();
            pt[0] = {pose_int(0), pose_int(1)};
        }

        MatrixXf s_body = scmap_body.block(y, x, h, w);

        if (s_body.maxCoeff() > TF_thresh) {
            s_body.maxCoeff(&maxloc_row, &maxloc_col);
            maxloc << maxloc_col + x, maxloc_row + y;
            offset << locref_body(maxloc(1), maxloc(0), 0), locref_body(maxloc(1), maxloc(0), 1);
            offset *= 7.2801f;
            MatrixXf pose = (maxloc + cst) * 8.0f + offset;
            pose_int = pose.cast<int>();
            pt[1] = {pose_int(0), pose_int(1)};
        }

        MatrixXf s_tail = scmap_tail.block(y, x, h, w);

        if (s_tail.maxCoeff() > TF_thresh) {
            s_tail.maxCoeff(&maxloc_row, &maxloc_col);
            maxloc << maxloc_col + x, maxloc_row + y;
            offset << locref_tail(maxloc(1), maxloc(0), 0), locref_tail(maxloc(1), maxloc(0), 1);
            offset *= 7.2801f;
            MatrixXf pose = (maxloc + cst) * 8.0f + offset;
            pose_int = pose.cast<int>();
            pt[2] = {pose_int(0), pose_int(1)};
        }
        /* TensorFlow (end) */

        if (pt[0].x != 0 && pt[0].y != 0){
            if (pt[1].x != 0 && pt[1].y != 0) {
                if (pt[2].x != 0 && pt[2].y != 0) {
                    cv::line(mat_img, pt[0], pt[1], SCALAR_WHITE, 1);
                    cv::line(mat_img, pt[1], pt[2], SCALAR_WHITE, 1);

                    // Grabs pixels along the line (pt[0], pt[1]) from 8-bit 3-channel image to the buffer
                    cv::LineIterator it(mat_img, pt[0], pt[1], 8);
                    std::vector<cv::Point> points(it.count);
                    for(int i = 0; i < it.count; i++, ++it) points[i] = it.pos();

                    WorldCoordinates(mat_img, points, p_pt);
                    std::vector<cv::Point>().swap(points);
                }
            }
        }

        //cv::Scalar color = obj_id_to_color(i.obj_id);
        if (MaestroFlag) { cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), SCALAR_PINK, 1);
        } else { cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), SCALAR_LIGHTGREEN, 1); }
    }

    if (current_det_fps >= 0 && current_cap_fps >= 0) {
        //std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
        std::string fps_str = "FPS: " + std::to_string(current_det_fps) + "   SessTime: " + std::to_string(realTime_now);
        putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
    }
}

void SessionApp::WorldCoordinates(cv::Mat& mat_img, std::vector<cv::Point>& px, cv::Point* p_pt) {

    rs2::depth_frame d_frame = depth_frame.as<rs2::depth_frame>();

    // Get depth value of either head or body
    float depth = 0.0f;
    std::vector<float> vec_depth;

    for (cv::Point& elem : px) {
        depth = d_frame.get_distance(elem.x, elem.y);
        if (!(depth < BASE || depth >  UPPERLIMIT)) {
            vec_depth.push_back(depth);
        }
    }

    if (vec_depth.size()) {
        // Get median
        std::nth_element(vec_depth.begin(), vec_depth.begin() + vec_depth.size()/2, vec_depth.end());
        depth = vec_depth[vec_depth.size()/2];
        old_depth = depth;
    } else {
        depth = old_depth; // In case failure in obtaining depth value
    }

    std::vector<float>().swap(vec_depth);

    // Distortion correction
    depth = depth * SLOPE + CONSTANT;

    // Color pixel with depth value to a 3D point (Assuming the depth value is the same for head, body, and tail)
    struct Coordinates coordinates = {};
    float pt[3] = {};
    float fpx[2] = {};

    // head part
    fpx[0] = static_cast<float>(p_pt[0].x);
    fpx[1] = static_cast<float>(p_pt[0].y);
    rs2_deproject_pixel_to_point(pt, &intrin_color, fpx, depth);

    coordinates.time = RealTime();
    coordinates.headX = Float2Int(pt[0] * 1000 - origin_pt.x);
    coordinates.headY = Float2Int(pt[1] * 1000 - origin_pt.y);
    coordinates.z = Float2Int((pt[2] - BASE) * 1000);

    if (!ValidateResponse_1(coordinates)) return;

    // body
    fpx[0] = static_cast<float>(p_pt[1].x);
    fpx[1] = static_cast<float>(p_pt[1].y);
    rs2_deproject_pixel_to_point(pt, &intrin_color, fpx, depth);
    coordinates.bodyX = Float2Int(pt[0] * 1000 - origin_pt.x);
    coordinates.bodyY = Float2Int(pt[1] * 1000 - origin_pt.y);

    // tail
    fpx[0] = static_cast<float>(p_pt[2].x);
    fpx[1] = static_cast<float>(p_pt[2].y);
    rs2_deproject_pixel_to_point(pt, &intrin_color, fpx, depth);
    coordinates.tailX = Float2Int(pt[0] * 1000 - origin_pt.x);
    coordinates.tailY = Float2Int(pt[1] * 1000 - origin_pt.y);

    // Display dots & the head value on color frame
    cv::circle(mat_img, p_pt[0], 3, SCALAR_RED, -1);
    cv::circle(mat_img, p_pt[1], 3, SCALAR_BLUE, -1);
    cv::circle(mat_img, p_pt[2], 3, SCALAR_GREEN, -1);
    std::string world_str = std::string(std::to_string(coordinates.headX) + ", " + std::to_string(coordinates.headY) + ", "
                                        + std::to_string(coordinates.z)).c_str();
    cv::putText(mat_img, world_str, p_pt[0], cv::FONT_HERSHEY_COMPLEX, 0.5, SCALAR_WHITE, 1, CV_AA);

    if (!ValidateResponse_2(coordinates)) return;
    if (!MaestroFlag) EvaluateResponse(coordinates);
}

void SessionApp::EvaluateResponse(Coordinates& coordinates) {

    for (int i = 1; i < 4; i++) {
        if (coordinates.headX > cCr[i].x_low && coordinates.headX < cCr[i].x_high &&
            coordinates.headY > cCr[i].y_low && coordinates.headY < cCr[i].y_high &&
            coordinates.z > cCr[i].z_low && coordinates.z < cCr[i].z_high) {
            if (lastResponse != i) {
                dvResponse[i]++;

                switch (i) {
                    case 1:
                    case 3:
                        std::cout << "Response " << i << ": " << dvResponse[i] << "\n";
                        break;
                }

                //respFlag[i] = true;
                lastResponse  = i; // New
                std::cout << "Status: " << lastResponse << "\n";

                char buf[4];
                sprintf(buf, "0%d", i);
                EventMarker(buf, coordinates);
            }
            break;
        }
    }
}

void SessionApp::ScheduleCheck(int side) {
//    switch (ivSchedule_X) {
//        case 2: // FR (Not used in this program)
//            tmpResponse++;
//            if (tmpResponse >= ivScheduleParameter_X) {
//                tmpResponse = 0;
//                dvReinforcer++;
//                //EventMarker("09");
//                std::cout << "Reinforcer: " << dvReinforcer << "\n";
//                MaestroFlag = true;
//            }
//            break;
//        case 3: // VI
//            if (SRarranged) {
//                // Added on 2021-8-30
//                FeederOnsetTime = get_dtime();
//
//                if (ivReversedSRLocation_X) { // Added on 2021-9-7
//                    if (seqSR == 0) {
//                        SRside = 1;
//                    } else {
//                        SRside = 0;
//                    }
//                } else { SRside = seqSR; }
//
//                if (!SRside) { // Added ! on 2021-9-4, replaced with SRside on 2021-9-7
//                    dvReinforcer_1++;
//                    EventMarker("10");
//                    std::cout << "Reinforcer_1 (Right): " << dvReinforcer_1 << "\n";
//                } else {
//                    dvReinforcer_2++;
//                    EventMarker("11");
//                    std::cout << "Reinforcer_2 (Left): " << dvReinforcer_2 << "\n";
//                }
//
//                MaestroFlag = true; // Moved from below for ending session correctly (2021-5-31)
//                dvReinforcer++;
//
//                lastResp = 0;
//
//                Value_Variable++;
//                Fleshler_Hoffman(ivScheduleParameter_X);
//                Period_Start = get_dtime() + SRduration;
//
//                //MaestroFlag = true;
//            }
//            break;
//    }
}

void SessionApp::EndSession() {

    // Create directories if they don't exist
    std::experimental::filesystem::path dir = "/home/tk/datafiles/" + FishID + "/rawdata/"; // Directory where data files will be saved
    if (!std::experimental::filesystem::is_directory(dir)) {
        std::experimental::filesystem::create_directories(dir);
    }

    // Count the number of files in a directory
    int filecount = file_count_native(dir.string());
    filecount++;
    std::string sess_num = std::to_string(filecount);

    // Create a raw data file
    std::string fname = std::string(FishID + "_" + sess_num + "_raw.txt").c_str();
    std::string dir_fname = std::string(dir.c_str() + fname).c_str();

    int len = static_cast<int> (dir_fname.length());
    char* pFilename = new char[len + 1];
    memcpy(pFilename, dir_fname.c_str(), len + 1);

    std::ofstream ofs1;
    ofs1.open(pFilename, std::ios::app); // Create or append a file

    GetDateTime(DateTime_end);

    ofs1 << "Start: " << DateTime_start;
    ofs1 << "End:   " << DateTime_end;
    ofs1 << "\n";
    ofs1 << "#Events: " << std::to_string(NumOfEvents) << "\n";
    ofs1 << "\n";
    ofs1 << "<DEPENDENT VARIABLES>\n";
    ofs1 << "Left compartment:   " << dvResponse[1] << "\n";
    ofs1 << "Center compartment: " << dvResponse[2] << "\n";
    ofs1 << "Right compartment:  " << dvResponse[3] << "\n";
    ofs1 << "Reinforcer:         " << dvReinforcer << "\n";
    ofs1 << "Reinforcer_1:       " << dvReinforcer_1 << "\n";
    ofs1 << "Reinforcer_2:       " << dvReinforcer_2 << "\n";
    ofs1 << "\n";
    ofs1 << "<INDEPENDENT VARIABLES>\n";
    ofs1 << "Schedule: 0 = EXT, 1 = VT, 2 = FR, 3 = VI\n";
    ofs1 << "Schedule:           " << ivSchedule_X << "\n";
    ofs1 << "ivMotionSide:       " << ivMotionSide << "\n";
    ofs1 << "OpenGLMotionSpeed:  " << OpenGLMotionSpeed << "\n";
    ofs1 << "TargetLocation_1:   " << ivTargetCorner_1_X << "\n";
    ofs1 << "TargetLocation_2:   " << ivTargetCorner_2_X << "\n";
    ofs1 << "ivPhase1:           " << ivPhase1_X << "\n";
    ofs1 << "ivPhase2:           " << ivPhase2_X << "\n";
    ofs1 << "ivPhase3:           " << ivPhase3_X << "\n";
    ofs1 << "ReturnTime:         " << ivPhase3_ReturnTime_X << "\n";

    ofs1 << "idlingCriterion:    " << idlingCriterion << "\n";
    ofs1 << "\n";
    ofs1 << "DEPTH_CORRECTION\n";
    ofs1 << "BASE:               " << BASE << "\n";
    ofs1 << "SLOPE:              " << SLOPE << "\n";
    ofs1 << "INTERCEPT:          " << INTERCEPT << "\n";
    ofs1 << "UPPERLIMIT:         " << UPPERLIMIT << "\n";
    ofs1 << "\n";
    ofs1 << "CORNER_CRITERIA\n";
    for (int i = 1; i < 4; ++i){
        ofs1 << cCr[i].x_low << ", " << cCr[i].x_high << ", "
                << cCr[i].y_low << ", " << cCr[i].y_high << ", "
                << cCr[i].z_low << ", " << cCr[i].z_high << "\n";
    }
    ofs1 << "\n";
    ofs1 << "<EVENT MARKERS>\n";
    ofs1 << "Event: During SR (No:0,R:1,L:2), Realtime, headX, headY, depth, bodyX, bodyY, tailX, tailY\n";
    ofs1 << "00: 3D coordinates\n";
    ofs1 << "01: Left compartment\n";
    ofs1 << "02: Center compartment\n";
    ofs1 << "03: Right compartment\n";
    ofs1 << "10: Reinforcer(1) onset\n";
    ofs1 << "11: Reinforcer(2) onset\n";
    ofs1 << "99: End of session\n";
    ofs1 << "\n";
    ofs1 << "<EVENTS>\n";
    for (std::string x : EventMarkers) ofs1 << x.c_str();
    ofs1 << "\n";
    ofs1 << "FINISH\n";
    ofs1 << "\n";

    ofs1.close();
    delete[] pFilename; // Release memory



    // Create a data file for 3D trajectory
    std::experimental::filesystem::path dir_trj = "/home/tk/datafiles/" + FishID + "/traj/";
    if (!std::experimental::filesystem::is_directory(dir_trj)) {
        std::experimental::filesystem::create_directories(dir_trj);
    }

    std::string fname_trj = std::string(FishID + "_" + sess_num + "_traj.csv").c_str();
    std::string dir_fname_trj = std::string(dir_trj.c_str() + fname_trj).c_str();

    int len_trj = static_cast<int> (dir_fname_trj.length());
    char* pFilename_trj = new char[len_trj + 1];
    memcpy(pFilename_trj, dir_fname_trj.c_str(), len_trj + 1);

    std::ofstream ofs2;
    ofs2.open(pFilename_trj, std::ios::app); // Create or append a file
    ofs2 << "x,y,z,color" << std::endl;

    //"Event: During SR, Realtime, headX, headY, bodyX, bodyY, tailX, tailY, depth, angle\n";
    // Reference: https://maku77.github.io/cpp/string/split.html
    for (std::string x : EventMarkers) {
        if (strstr(x.c_str(), "00: ") != NULL) {
            std::string subx = x.substr(4);
            std::istringstream iss(subx);
            char delim = ',';
            std::string tmp;
            std::vector<std::string> res;
            while (getline(iss, tmp, delim)) res.push_back(tmp);
            ofs2 << res[2] << "," << res[3] << "," << res[4] << "," << res[0] << "\n";
            std::vector<std::string>().swap(res);
        }
    }

    ofs2.close();
    delete[] pFilename_trj; // Release memory

    std::vector<int>().swap(sequence);
}

int SessionApp::file_count_native(const std::string& dir) {
    DIR *dp;
    int i = 0;
    struct dirent *ep;
    dp = opendir (dir.c_str());

    if (dp != NULL) {
        while ((ep = readdir(dp)) != NULL) {
            if (ep->d_type == DT_REG) i++;
        }
        closedir(dp);
    }
    else perror ("Couldn't open the directory");

    return i;
}

void SessionApp::EventMarker(const char* EventType) {
    char tmpStr[30] = {};
    sprintf(tmpStr, "%s: %d\n", EventType, RealTime());
    std::string str(tmpStr);
    EventMarkers.push_back(str);
    NumOfEvents++;
}

void SessionApp::EventMarker(const char* EventType, const Coordinates& coordinates) {
    char tmpStr[50] = {};
    if (!MaestroFlag) {
        sprintf(tmpStr, "%s: %d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                EventType, 0, coordinates.time,
                coordinates.headX, coordinates.headY, coordinates.z,
                coordinates.bodyX, coordinates.bodyY,
                coordinates.tailX, coordinates.tailY);
    }
    else {
        if (!SRside) {
            // "v_frame1_Right"
            sprintf(tmpStr, "%s: %d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                    EventType, 1, coordinates.time,
                    coordinates.headX, coordinates.headY, coordinates.z,
                    coordinates.bodyX, coordinates.bodyY,
                    coordinates.tailX, coordinates.tailY);
        } else {
            // "v_frame2_Left"
            sprintf(tmpStr, "%s: %d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                    EventType, 2, coordinates.time,
                    coordinates.headX, coordinates.headY, coordinates.z,
                    coordinates.bodyX, coordinates.bodyY,
                    coordinates.tailX, coordinates.tailY);
        }
    }

    std::string str(tmpStr);
    EventMarkers.push_back(str);
    NumOfEvents++;
}

int SessionApp::ValidateResponse_1(const Coordinates& coordinates) {

    if (coordinates.headX >= AQUARIUM_minX && coordinates.headX < AQUARIUM_X - AQUARIUM_minX) // Check if the coordinates are within a range of aquarium
        if (coordinates.headY >= AQUARIUM_minY && coordinates.headY < AQUARIUM_Y - AQUARIUM_minY)
            if (coordinates.z >= AQUARIUM_minZ && coordinates.z < AQUARIUM_Z - AQUARIUM_minZ)
                return 1;
    return 0;
}

int SessionApp::ValidateResponse_2(const Coordinates& coordinates) {

    deqCoordinates.push_front(coordinates);
    int n = deqCoordinates.size();
    if (n < 5) return 0;

    // Identify possible false positives
    bool poss_hit = false;
    if (abs(deqCoordinates[0].headX - deqCoordinates[1].headX) < diffCr.x)
        if (abs(deqCoordinates[1].headX - deqCoordinates[2].headX) < diffCr.x)
            if (abs(deqCoordinates[2].headX - deqCoordinates[3].headX) < diffCr.x)
                if (abs(deqCoordinates[3].headX - deqCoordinates[4].headX) < diffCr.x)
                    if (abs(deqCoordinates[0].headY - deqCoordinates[1].headY) < diffCr.y)
                        if (abs(deqCoordinates[1].headY - deqCoordinates[2].headY) < diffCr.y)
                            if (abs(deqCoordinates[2].headY - deqCoordinates[3].headY) < diffCr.y)
                                if (abs(deqCoordinates[3].headY - deqCoordinates[4].headY) < diffCr.y)
                                    if (abs(deqCoordinates[0].z - deqCoordinates[1].z) < diffCr.z)
                                        if (abs(deqCoordinates[1].z - deqCoordinates[2].z) < diffCr.z)
                                            if (abs(deqCoordinates[2].z - deqCoordinates[3].z) < diffCr.z)
                                                if (abs(deqCoordinates[3].z - deqCoordinates[4].z) < diffCr.z)
                                                    poss_hit = true;

    if (poss_hit) {
        // 0. Reset flags
        idlingFlag = false;
        automaticMotion = false;

        // 1. Convert coordinates to Pixel (Moving average for smoothier motion on OpenGL)
        const int count = 5;
        int sum_x = 0, sum_y = 0;
        for (int i = 0; i < count; ++i) {
            sum_x += deqCoordinates[i].headY;
            sum_y += deqCoordinates[i].z;
        }
        float mean_x = (float)sum_x / count;
        float mean_y = (float)sum_y / count;
        int x = (int)((mean_x / (float)AQUARIUM_Y) * (float)V_WIDTH);
        int y = (int)((mean_y / (float)AQUARIUM_Z) * (float)V_HEIGHT);

        // 2. Send data to OpenGL
        switch (ivMotionSide) {
            case 0: // Right
            case 5: // Right
            case 8:
            case 10:
            case 12:
                extern_target0.x = x;
                extern_target0.y = y;

                extern_target1.x = V_WIDTH - x;
                extern_target1.y = y;
                break;
            case 1: // Left
            case 6: // Left
            case 7:
            case 9:
            case 11:
                extern_target0.x = V_WIDTH - x;
                extern_target0.y = y;

                extern_target1.x = x;
                extern_target1.y = y;
                break;
        }


        if (prev_memoryflag) {
            EventMarker("00", deqCoordinates[4]);
            EventMarker("00", deqCoordinates[3]);
            EventMarker("00", deqCoordinates[2]);
            EventMarker("00", deqCoordinates[1]);
            prev_memoryflag = false;
        }
        EventMarker("00", deqCoordinates[0]);
        deqCoordinates.pop_back();
        return 1;
    } else {
        // Temporarily save the coordinates without EventMarker()
        deqCoordinates.pop_back();
        prev_memoryflag = true;
        return 0;
    }
}

int SessionApp::SessionTime() {
    return Double2Int(get_dtime() - Session_StartTime - (SRduration * dvReinforcer));
}

int SessionApp::RealTime() {
    return Double2Int(get_dtime() - Session_StartTime);
}

void SessionApp::GetDateTime(char *DateTime) {
    time_t timer;
    struct tm* local;
    timer = time(NULL);
    local = localtime(&timer);
    sprintf(DateTime, "%d/%d/%d %d:%d:%d\n", local->tm_year + 1900, local->tm_mon + 1, local->tm_mday, local->tm_hour, local->tm_min, local->tm_sec);
}

void SessionApp::saveOrigin() {
    // Create directories if they don't exist
    std::experimental::filesystem::path dir = "/home/tk/datafiles/origin/" + FishID + "/"; // Directory where the origin of 3D coordinates will be saved
    if (!std::experimental::filesystem::is_directory(dir)){
        std::experimental::filesystem::create_directories(dir);
    }

    std::string fname = std::string("Origin.txt").c_str();
    std::string dir_fname = std::string(dir.c_str() + fname).c_str();

    int len = static_cast<int> (dir_fname.length());
    char* pFname = new char[len + 1];
    memcpy(pFname, dir_fname.c_str(), len + 1);

    std::ofstream ofs(pFname);
    ofs << origin_pt.x << std::endl;
    ofs << origin_pt.y << std::endl;
    ofs << origin_pt.z << std::endl;
    ofs.close();
    delete[] pFname; // Release memory
}

void SessionApp::AdjustOrigin() {
    rs2::depth_frame d_frame = depth_frame.as<rs2::depth_frame>();

    float pt[3] = {};
    float px_c[2] = { origin_px.x, origin_px.y };
    float depth = d_frame.get_distance(Float2Int(origin_px.x), Float2Int(origin_px.y));

    int depth_i = Float2Int(depth * 1000);
    if (depth_i) {
        depth = depth * SLOPE + CONSTANT; // Distortion correction

        // Color pixel with depth value to a 3D point
        rs2_deproject_pixel_to_point(pt, &intrin_color, px_c, depth);

        origin_pt.x = pt[0] * 1000;
        origin_pt.y = pt[1] * 1000;
        origin_pt.z = pt[2] * 1000;

        std::cout << "Origin set\n";
        std::cout << "  Color (in pixel): X = " << origin_px.x << ", Y = " << origin_px.y << "\n";
        std::cout << "  World (in mm): X = " << origin_pt.x << ", Y = " << origin_pt.y << ", Z = " << origin_pt.z << "\n\n";
    }
    else { std::cout << "Failed to get depth\n"; }

    MouseInEffect = false;
}

void SessionApp::checkOriginFile() {
    // Create directories if they don't exist
    std::experimental::filesystem::path dir = "/home/tk/datafiles/origin/" + FishID + "/"; // Directory where the origin of 3D coordinates is saved
    if (!std::experimental::filesystem::is_directory(dir)){
        std::experimental::filesystem::create_directories(dir);
    }

    std::string fname = std::string("Origin.txt").c_str();
    std::string dir_fname = std::string(dir.c_str() + fname).c_str();

    int len = static_cast<int> (dir_fname.length());
    char* pFname = new char[len + 1];
    memcpy(pFname, dir_fname.c_str(), len + 1);

    std::ifstream ifs(pFname);
    if (ifs) {
        std::string dow_x, dow_y, dow_z;

        getline(ifs, dow_x);
        getline(ifs, dow_y);
        getline(ifs, dow_z);

        origin_pt.x = std::stof(dow_x);
        origin_pt.y = std::stof(dow_y);
        origin_pt.z = std::stof(dow_z);

        ifs.close();
    }
    else {
        std::ofstream ofs(pFname);
        ofs << "0\n" << "0\n" << "0\n";
        ofs.close();
    }

    delete[] pFname; // Release memory
}

std::vector<std::string> SessionApp::objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for(std::string line; getline(file, line);) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}

void SessionApp::updateFrame() {

    // Wait for next set of frames from the camera
    frames = pipeRS.wait_for_frames();

    // Standard streams (Disable the next two codes when enabling coordinate mapping)
    //color_frame = frames.get_color_frame();
    //depth_frame = frames.get_depth_frame();

    // Streams with coordinate mapping (Align Depth to Color)
    rs2::align align(rs2_stream::RS2_STREAM_COLOR);
    rs2::frameset aligned_frames = align.process(frames);
    color_frame = frames.get_color_frame();
    depth_frame = aligned_frames.get_depth_frame();
    rs2::frame colorized_depth_frame = color_map.colorize(depth_frame);

    colorFrame = cv::Mat(cv::Size(COLOR_WIDTH, COLOR_HEIGHT), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
    depthFrame = cv::Mat(cv::Size(DEPTH_WIDTH, DEPTH_HEIGHT), CV_8UC3, (void*)colorized_depth_frame.get_data(), cv::Mat::AUTO_STEP);
}

void SessionApp::mouseCallback(int event, int x, int y, int flags, void* userdata) {
    auto pThis = (SessionApp*)userdata;
    pThis->mouseCallback(event, x, y, flags);
}

void SessionApp::mouseCallback(int event, int x, int y, int flags) {
    if (event == CV_EVENT_LBUTTONDOWN) {
        origin_px.x = static_cast<float> (x);
        origin_px.y = static_cast<float> (y);
        MouseInEffect = true;
    }
}

void SessionApp::initRS(){
// Make a list of RealSense cameras connected to PC
    rs2::context ctx;
    const rs2::device_list dev_list = ctx.query_devices();
    int dev_count = 0;

    if (dev_list.size() > 0) {
        for (const rs2::device& dev : dev_list) {
            // Get the serial number of RealSense camera
            const std::string serial = dev.get_info(rs2_camera_info::RS2_CAMERA_INFO_SERIAL_NUMBER);

            if (serial == SerialNumber) {
                dev_count++;

                // Enable Color & Depth streams
                cfg.enable_device(serial);
                cfg.enable_stream(RS2_STREAM_COLOR, COLOR_WIDTH, COLOR_HEIGHT, RS2_FORMAT_BGR8, COLOR_FPS);
                cfg.enable_stream(RS2_STREAM_DEPTH, DEPTH_WIDTH, DEPTH_HEIGHT, RS2_FORMAT_Z16, DEPTH_FPS);

                // Added on 2019-6-1; Removed on 2020-1-9 (10am)
                auto depthSensor = dev.first<rs2::depth_sensor>();
                if (depthSensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
                	depthSensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.f); // Currently turned off (Depth is more accurate without the internal IR although depthFrame look ugly)
                }

                // Start the pipeline for the streams
                profile = pipeRS.start(cfg);

                std::cout << "\n";

                // Get intrinsic & extrinsic parameters of depth camera
                rs2::video_stream_profile stream_color = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
                intrin_color = stream_color.get_intrinsics();
                rs2_distortion model_color = intrin_color.model;
                std::cout << "Intrinsics: Color Camera\n";
                std::cout << "Principal Point         : " << intrin_color.ppx << ", " << intrin_color.ppy << "\n";
                std::cout << "Focal Length            : " << intrin_color.fx << ", " << intrin_color.fy << "\n";
                std::cout << "Distortion Model        : " << model_color << "\n";
                std::cout << "Distortion Coefficients : [" << intrin_color.coeffs[0] << "," << intrin_color.coeffs[1] << "," <<
                          intrin_color.coeffs[2] << "," << intrin_color.coeffs[3] << "," << intrin_color.coeffs[4] << "]" << "\n";
                std::cout << "\n";

                rs2::video_stream_profile stream_depth = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
                intrin_depth = stream_depth.get_intrinsics();
                rs2_distortion model_depth = intrin_depth.model;
                std::cout << "Intrinsics: Depth Camera\n";
                std::cout << "Principal Point         : " << intrin_depth.ppx << ", " << intrin_depth.ppy << "\n";
                std::cout << "Focal Length            : " << intrin_depth.fx << ", " << intrin_depth.fy << "\n";
                std::cout << "Distortion Model        : " << model_depth << "\n";
                std::cout << "Distortion Coefficients : [" << intrin_depth.coeffs[0] << "," << intrin_depth.coeffs[1] << "," <<
                          intrin_depth.coeffs[2] << "," << intrin_depth.coeffs[3] << "," << intrin_depth.coeffs[4] << "]" << "\n";
                std::cout << "\n";

                extrin_c2d = stream_color.get_extrinsics_to(stream_depth);
                std::cout << "Extrinsics: Color to Depth\n";
                std::cout << "Translation Vector : [" << extrin_c2d.translation[0] << "," << extrin_c2d.translation[1] << "," << extrin_c2d.translation[2] << "]\n";
                std::cout << "Rotation Matrix    : [" << extrin_c2d.rotation[0] << "," << extrin_c2d.rotation[3] << "," << extrin_c2d.rotation[6] << "]\n";
                std::cout << "                   : [" << extrin_c2d.rotation[1] << "," << extrin_c2d.rotation[4] << "," << extrin_c2d.rotation[7] << "]\n";
                std::cout << "                   : [" << extrin_c2d.rotation[2] << "," << extrin_c2d.rotation[5] << "," << extrin_c2d.rotation[8] << "]\n";
                std::cout << "\n";

                extrin_d2c = stream_depth.get_extrinsics_to(stream_color);
                std::cout << "Extrinsics: Depth to Color\n";
                std::cout << "Translation Vector : [" << extrin_d2c.translation[0] << "," << extrin_d2c.translation[1] << "," << extrin_d2c.translation[2] << "]\n";
                std::cout << "Rotation Matrix    : [" << extrin_d2c.rotation[0] << "," << extrin_d2c.rotation[3] << "," << extrin_d2c.rotation[6] << "]\n";
                std::cout << "                   : [" << extrin_d2c.rotation[1] << "," << extrin_d2c.rotation[4] << "," << extrin_d2c.rotation[7] << "]\n";
                std::cout << "                   : [" << extrin_d2c.rotation[2] << "," << extrin_d2c.rotation[5] << "," << extrin_d2c.rotation[8] << "]\n";
                std::cout << "\n";
            }
        }

        if (dev_count == 0) {
            dev_flag = false;
        }
    }
    else {
        dev_flag = false;
    }
}

void SessionApp::Fleshler_Hoffman(int Parameter) {
    if (Value_Variable == n) Value_Variable = 0;

    if (Value_Variable == 0) {
        v = Parameter * 1000;

        int b = 0;
        for (b = 0; b <= n; b++) {
            rd[b] = 0;
            vi[b] = 0;
        }

        int i = 0;
        for (i = 1; i <= n; i++) {
            if (i == n) { vi[i] = Double2Int(v * (1 + log(n))); }
            else { vi[i] = Double2Int(v * (1 + (log(n)) + (n - i) * (log(n - i)) - (n - i + 1) * log(n - i + 1))); }

            do {
                order = XorShift() % n;
            } while (rd[order] != 0);

            rd[order] = vi[i];
        }

        int a = 0;
        for (a = 0; a <= n; a++) Sumxser = Sumxser + rd[a];

        if (Sumxser != (v * n)) rd[0] = rd[0] + ((v * n) - Sumxser);
        Sumxser = 0;
    }
}

void SessionApp::QuasiRandom_video(std::vector<int>& seq, int max, int segment) {

    int seg = 0;
    int adj = 0;
    int* arr;
    arr = (int*)malloc(sizeof(int) * max);
    memset(arr, -1, sizeof(int) * max);

    int loop_count = max / segment;
    for (int i = 0; i < loop_count; ++i) {
        adj = i * segment;
        for (int j = 0; j < segment; ++j) {
            do {
                seg = XorShift() % segment;
            } while (arr[seg + adj] != -1);
            arr[seg + adj] = j;
        }
    }
    copy(arr, arr + max, seq.begin());
    free(arr);
}

void SessionApp::QuasiRandom(std::vector<int>& seq, int max, int ratio) {

    int n = ratio / 2;
    int blockSize = max / 2;
    float prob = 0.f;
    if (ratio) {
        prob = (float)(max / ratio) * 1.1; // 1.1 to increase probability a bit for when ratio is greater than 20
    } else {
        prob = (float)(max / 0.001);
    }

    int maxRep = max / blockSize;

    int rep = 0;
    int offset = 0;

    int type = 0;
    int count = 0;
    int type0 = 0;

    for (int i = 0; i < maxRep; ++i) {
        while(1) {
            type = (int)(prob * (XorShift() / (float)ULONG_MAX));
            if (type != 0) {
                type = 1;
            }

            seq[offset + count] = type;
            count++;

            if (type == 0) {type0++;};
            if (count >= blockSize) {
                //std::cout << type0 << "\n";
                if (type0 == n) {
                    rep++;
                    offset = blockSize * rep;
                    type = 0;
                    count = 0;
                    type0 = 0;
                    break;
                }

                for (int j = 0; j < blockSize; ++j) {
                    seq[offset + j] = 0;
                }

                type = 0;
                count = 0;
                type0 = 0;
            }
        }
    }
}

void SessionApp::init_XorShift() {
    unsigned long s = clock();
    unsigned long i = 0;
    for (i = 0; i < 4; ++i) {
        seed[i] = s = 1812433253U * (s ^ (s >> 30)) + i;
    }
}

unsigned long SessionApp::XorShift() {
    unsigned long t = (seed[0] ^ (seed[0] << 11));
    seed[0] = seed[1];
    seed[1] = seed[2];
    seed[2] = seed[3];

    return (seed[3] = (seed[3] ^ (seed[3] >> 19)) ^ (t ^ (t >> 8)));
}

SessionApp::SessionApp(){
    init_XorShift(); // Initialize XorShift RNG

    ivSchedule_X = ivSchedule;
    ivScheduleParameter_X = ivScheduleParameter; // Irrelevant
    ivReversedSRLocation_X = ivReversedSRLocation; // Irrelevant
    ivPhase1_X = ivPhase1;
    ivPhase2_X = ivPhase2;
    ivPhase3_X = ivPhase3;
    ivPhase3_ReturnTime_X = ivPhase3_ReturnTime;
    phaseFlag = 1; // Start with 1

    ivSRprop_X = ivSRprop; // Irrelevant
    ivMaxSR_X = ivMaxSR; // Irrelevant
    ivTargetCorner_1_X = ivTargetCorner_1;
    ivTargetCorner_2_X = ivTargetCorner_2;


    cCr[1] = {                 AQUARIUM_minX, (int)((1.0 * AQUARIUM_X) / 3), AQUARIUM_minY, AQUARIUM_Y - AQUARIUM_minY, AQUARIUM_minZ, AQUARIUM_Z - AQUARIUM_minZ }; // Left compartment
    cCr[2] = { (int)((1.0 * AQUARIUM_X) / 3), (int)((2.0 * AQUARIUM_X) / 3), AQUARIUM_minY, AQUARIUM_Y - AQUARIUM_minY, AQUARIUM_minZ, AQUARIUM_Z - AQUARIUM_minZ }; // Center compartment
    cCr[3] = { (int)((2.0 * AQUARIUM_X) / 3),    AQUARIUM_X - AQUARIUM_minX, AQUARIUM_minY, AQUARIUM_Y - AQUARIUM_minY, AQUARIUM_minZ, AQUARIUM_Z - AQUARIUM_minZ }; // Right compartment

    MaestroFlag = false;
    EndMaestroFlag = false;
}

SessionApp::~SessionApp(){
    cv::destroyAllWindows();
    deqCoordinates.clear();
    deqCoordinates.shrink_to_fit();
}

