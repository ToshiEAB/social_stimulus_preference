#ifndef MODELAPP_H
#define MODELAPP_H


#include <math.h>
#include <time.h>

#include <GL/freeglut.h>
#include <GL/glext.h>

#include <fstream>
#include <map>
#include <vector>


#include <iostream>
#include <atomic>
#include <thread>
#include <math.h>
#include <bits/stdc++.h> // For pi
#include <string>
#include "extern.h"

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "../Library/loadpng.h"
#include "../Library/process_image.h"
#include "../Library/gl_texture.h"


const int WIDTH = 1024;
const int HEIGHT = 600;

const float x_min = -19.0f;// Left
const float x_max = 27.5f; // Right
const float y_min = -14.0f;// Bottom
const float y_max = 11.5f;// Top
const float x_range = x_max - x_min;
const float y_range = y_max - y_min;
const float x_ratio = x_range / (float)WIDTH;
const float y_ratio = y_range / (float)HEIGHT;
const float fishWidth = 8.4f;
const float constAdj_y = fishWidth / 180.f;

int windowID[2];

std::string vfile = "../video/openGL_ver10.mp4";
cv::VideoCapture vcap;

struct measures {
    std::atomic<int> x;
    std::atomic<int> y;
    std::atomic<float> pos_x;
    std::atomic<float> pos_y;
    std::atomic<float> pos_z;
    measures() : x(0), y(0), pos_x(0.f), pos_y(0.f), pos_z(0.f) {}
};
extern measures extern_target0;
extern measures extern_target1;

class Model {
private:
    static int count_char(std::string &str, char ch) {
        int c = 0;
        int length = str.length() - 1;
        for (int i = 0; i < length; i++) {
            if (str[i] == ch)
                c++;
        }
        return c;
    }

    static bool has_double_slash(std::string &str) {
        int length = str.length() - 2;
        for (int i = 0; i < length; i++) {
            if (str[i] == '/' && str[i + 1] == '/')
                return true;
        }
        return false;
    }

    class Material {
    public:
        float *ambient;
        float *diffuse;
        float *specular;
        GLuint texture;

        Material(float *ambient, float *diffuse, float *specular) {
            this->ambient = ambient;
            this->diffuse = diffuse;
            this->specular = specular;
            this->texture = 0;
        }
    };

    class Face {
    public:
        int edge;
        int *vertices;
        int *texcoords;
        int normal;

        Face(int edge, int *vertices, int *texcoords, int normal = -1) {
            this->edge = edge;
            this->vertices = vertices;
            this->texcoords = texcoords;
            this->normal = normal;
        }
    };

    std::string prefix;
    std::vector<Material> materials;
    std::map<std::string, int> map_material;

    std::vector<float *> vertices;
    std::vector<float *> texcoords;
    std::vector<float *> normals;
    std::vector<Face> faces;

    GLuint list;

    void load_material(const char *filename) {
        std::string line;
        std::vector<std::string> lines;
        std::ifstream in(filename);
        if (!in.is_open()) {
            printf("Cannot load material %s\n", filename);
            return;
        }

        while (!in.eof()) {
            std::getline(in, line);
            lines.push_back(line);
        }
        in.close();

        Material *m;
        int count_material = 0;
        char str[40];
        std::string material;
        float *a, *d, *s;

        for (std::string &line : lines) {
            if (line[0] == 'n' && line[1] == 'e') {
                sscanf(line.c_str(), "newmtl %s", str);
                material = str;
                map_material[material] = count_material;
                count_material++;
                a = new float[4]{0.2f, 0.2f, 0.2f, 1.0f};
                d = new float[4]{0.8f, 0.8f, 0.8f, 1.0f};
                s = new float[4]{0.0f, 0.0f, 0.0f, 1.0f};
                materials.push_back(Material(a, d, s));
                m = &materials[materials.size() - 1];
            } else if (line[0] == 'K') {
                switch (line[1]) {
                    case 'a':
                        sscanf(line.c_str(), "Ka %f %f %f", &a[0], &a[1], &a[2]);
                        break;
                    case 'd':
                        sscanf(line.c_str(), "Kd %f %f %f", &d[0], &d[1], &d[2]);
                        break;
                    case 's':
                        sscanf(line.c_str(), "Ks %f %f %f", &s[0], &s[1], &s[2]);
                        break;
                }
            } else if (line[0] == 'm' && line[1] == 'a') {
                sscanf(line.c_str(), "map_Kd %s", str);
                std::string file = prefix + str;
                Image img;
                Load_Texture_Swap(&img, file.c_str());
                glGenTextures(1, &(m->texture));
                glBindTexture(GL_TEXTURE_2D, m->texture);
                glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.w, img.h, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.img);
                glBindTexture(GL_TEXTURE_2D, 0);
                Delete_Image(&img);
            }
        }
    }

    void add_face_3v(std::string &line) {
        int v0, v1, v2;
        sscanf(line.c_str(), "f %d %d %d", &v0, &v1, &v2);
        int *v = new int[3]{v0 - 1, v1 - 1, v2 - 1};
        faces.push_back(Face(3, v, NULL));
    }

    void add_face_3vt(std::string &line) {
        int v0, v1, v2, t0, t1, t2;
        sscanf(line.c_str(), "f %d/%d %d/%d %d/%d", &v0, &t0, &v1, &t1, &v2, &t2);
        int *v = new int[3]{v0 - 1, v1 - 1, v2 - 1};
        int *t = new int[3]{t0 - 1, t1 - 1, t2 - 1};
        faces.push_back(Face(3, v, t));
    }

    void add_face_3vn(std::string &line) {
        int v0, v1, v2, n;
        sscanf(line.c_str(), "f %d//%d %d//%d %d//%d", &v0, &n, &v1, &n, &v2, &n);
        int *v = new int[3]{v0 - 1, v1 - 1, v2 - 1};
        faces.push_back(Face(3, v, NULL, n - 1));
    }

    void add_face_3vtn(std::string &line) {
        int v0, v1, v2, t0, t1, t2, n;
        sscanf(line.c_str(), "f %d/%d/%d %d/%d/%d %d/%d/%d", &v0, &t0, &n, &v1, &t1, &n, &v2, &t2, &n);
        int *v = new int[3]{v0 - 1, v1 - 1, v2 - 1};
        int *t = new int[3]{t0 - 1, t1 - 1, t2 - 1};
        faces.push_back(Face(3, v, t, n - 1));
    }

    void add_face_4v(std::string &line) {
        int v0, v1, v2, v3;
        sscanf(line.c_str(), "f %d %d %d %d", &v0, &v1, &v2, &v3);
        int *v = new int[4]{v0 - 1, v1 - 1, v2 - 1, v3 - 1};
        faces.push_back(Face(4, v, NULL));
    }

    void add_face_4vt(std::string &line) {
        int v0, v1, v2, v3, t0, t1, t2, t3;
        sscanf(line.c_str(), "f %d/%d %d/%d %d/%d %d/%d", &v0, &t0, &v1, &t1, &v2, &t2, &v3, &t3);
        int *v = new int[4]{v0 - 1, v1 - 1, v2 - 1, v3 - 1};
        int *t = new int[4]{t0 - 1, t1 - 1, t2 - 1, t3 - 1};
        faces.push_back(Face(4, v, t));
    }

    void add_face_4vn(std::string &line) {
        int v0, v1, v2, v3, n;
        sscanf(line.c_str(), "f %d//%d %d//%d %d//%d %d//%d", &v0, &n, &v1, &n, &v2, &n, &v3, &n);
        int *v = new int[4]{v0 - 1, v1 - 1, v2 - 1, v3 - 1};
        faces.push_back(Face(4, v, NULL, n - 1));
    }

    void add_face_4vtn(std::string &line) {
        int v0, v1, v2, v3, t0, t1, t2, t3, n;
        sscanf(line.c_str(), "f %d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d", &v0, &t0, &n, &v1, &t1, &n, &v2, &t2, &n, &v3,
               &t3, &n);
        int *v = new int[4]{v0 - 1, v1 - 1, v2 - 1, v3 - 1};
        int *t = new int[4]{t0 - 1, t1 - 1, t2 - 1, t3 - 1};
        faces.push_back(Face(4, v, t, n - 1));
    }

public:
    float pos_x, pos_y, pos_z;

    void load(const char *filename) {
        std::string tmp = filename;
        prefix = "";
        int n = tmp.find_last_of('/') + 1;
        if (n > 0)
            prefix = tmp.substr(0, n);

        std::string line;
        std::vector<std::string> lines;
        std::ifstream in(filename);
        if (!in.is_open()) {
            printf("Cannot load model %s\n", filename);
            return;
        }

        while (!in.eof()) {
            std::getline(in, line);
            lines.push_back(line);
        }
        in.close();

        float a, b, c;
        char str[40];

        pos_x = pos_y = 0.0f;

        float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;

        for (std::string &line : lines) {
            if (line[0] == 'v') {
                if (line[1] == ' ') {
                    sscanf(line.c_str(), "v %f %f %f", &a, &b, &c);
                    if (a > 0.0f)
                        sum_x += a;
                    else
                        sum_x -= a;
                    if (b > 0.0f)
                        sum_y += b;
                    else
                        sum_y -= b;
                    if (c > 0.0f)
                        sum_z += c;
                    else
                        sum_z -= c;
                    pos_x += a;
                    pos_y += b;
                    vertices.push_back(new float[3]{a, b, c});
                } else if (line[1] == 't') {
                    sscanf(line.c_str(), "vt %f %f", &a, &b);
                    texcoords.push_back(new float[2]{a, b});
                } else {
                    sscanf(line.c_str(), "vn %f %f %f", &a, &b, &c);
                    normals.push_back(new float[3]{a, b, c});
                }
            } else if (line[0] == 'f') {
                int edge = count_char(line, ' ');
                int count_slash = count_char(line, '/');
                if (count_slash == 0) {
                    if (edge == 3)
                        add_face_3v(line);
                    else
                        add_face_4v(line);
                } else if (count_slash == edge) {
                    if (edge == 3)
                        add_face_3vt(line);
                    else
                        add_face_4vt(line);
                } else if (count_slash == edge * 2) {
                    if (has_double_slash(line)) {
                        if (edge == 3)
                            add_face_3vn(line);
                        else
                            add_face_4vn(line);
                    } else {
                        if (edge == 3)
                            add_face_3vtn(line);
                        else
                            add_face_4vtn(line);
                    }
                }
            } else if (line[0] == 'm' && line[1] == 't') {
                sscanf(line.c_str(), "mtllib %s", &str);
                std::string file = prefix + str;
                load_material(file.c_str());
            } else if (line[0] == 'u' && line[1] == 's') {
                sscanf(line.c_str(), "usemtl %s", &str);
                std::string material = str;
                if (map_material.find(material) != map_material.end())
                    faces.push_back(Face(-1, NULL, NULL, map_material[material]));
            }
        }

        bool has_texcoord = false;

        list = glGenLists(1);
        glNewList(list, GL_COMPILE);
        for (Face &face : faces) {
            if (face.edge == -1) {
                has_texcoord = false;
                glLightfv(GL_LIGHT0, GL_AMBIENT, materials[face.normal].ambient);
                glLightfv(GL_LIGHT0, GL_DIFFUSE, materials[face.normal].diffuse);
                glLightfv(GL_LIGHT0, GL_SPECULAR, materials[face.normal].specular);
                if (materials[face.normal].texture != 0) {
                    has_texcoord = true;
                    glBindTexture(GL_TEXTURE_2D, materials[face.normal].texture);
                }
                continue;
            }
            if (face.normal != -1)
                glNormal3fv(normals[face.normal]);
            else
                glDisable(GL_LIGHTING);
            if (has_texcoord) {
                glBegin(GL_POLYGON);
                for (int i = 0; i < face.edge; i++) {
                    glTexCoord2fv(texcoords[face.texcoords[i]]);
                    glVertex3fv(vertices[face.vertices[i]]);
                }
                glEnd();
            } else {
                glBegin(GL_POLYGON);
                for (int i = 0; i < face.edge; i++)
                    glVertex3fv(vertices[face.vertices[i]]);
                glEnd();
            }
            if (face.normal == -1)
                glEnable(GL_LIGHTING);
        }
        glEndList();

//        printf("Model: %s\n", filename);
//        printf("Vertices: %d\n", vertices.size());
//        printf("Texcoords: %d\n", texcoords.size());
//        printf("Normals: %d\n", normals.size());
//        printf("Faces: %d\n", faces.size());
//        printf("Materials: %d\n", materials.size());

        sum_x /= vertices.size();
        sum_y /= vertices.size();
        sum_z /= vertices.size();
        pos_x /= vertices.size();
        pos_x = -pos_x;
        pos_y /= vertices.size();
        pos_y = -pos_y;
        pos_z = -sqrt(sum_x * sum_x + sum_y * sum_y + sum_z * sum_z) * 15;

//        printf("Pos_X: %f\n", pos_x);
//        printf("Pos_Y: %f\n", pos_y);
//        printf("Pos_Z: %f\n", pos_z);

        for (Material &material : materials) {
            delete material.ambient;
            delete material.diffuse;
            delete material.specular;
        }

        materials.clear();
        map_material.clear();

        for (float *f : vertices)
            delete f;
        vertices.clear();
        for (float *f : texcoords)
            delete f;
        texcoords.clear();
        for (float *f : normals)
            delete f;
        normals.clear();
        faces.clear();
    }

    void draw() { glCallList(list); }
};

// There must be two different class for callbacks (Just copy) because the memory of static variables are shared
class callbacks0 {
public:
    std::string model_name = "../animation_models/mm_frame.obj";

    inline static bool initFlag = true;

    // *** Changed
    // 0 = Being chased (Works), 1 = Chasing ver.1 (Works), 2 = Centerizing motion (Works), 3 = Centerizing + WIDTH (Works),
    // 4 = Centerizing + WIDTH + No X (Works), 5 = Chasing ver.2 (Works), 6 = Chasing ver.3
    inline static int motionType = 0;

    // Common
    inline static const int motionThresh = 16;
    inline static int callbackID = 0;

    // 1: Chasing ver.1
    inline static const int chaseDist = 200; // Should be greater than 180
    inline static int awayPixel = 0;
    inline static float awayFloat = 0.f;

    // 2 & 3 & 4: Centerizing
    inline static int centMotionDirection = 0; // 0 = Right, 1 = Left
    inline static const int centMotionSpeed = 30;//15;
    inline static const int centMotionRange = 350;
    inline static const int outRange = 180;
    inline static int idleTarget_x = 0;
    inline static int lastX = 0, lastY = 0;

    // 5: Chasing ver.2
    inline static const int nearX = 350, nearY = 150;

    inline static measures target;

    // Must be inline for setting variables at the time of initialization
    inline static Model model;

    inline static int POS_X, POS_Y;
    inline static int INTERVAL = 15;

    inline static GLfloat light_pos[] = {-10.0f, 10.0f, 100.00f, 1.0f};

    inline static float pos_x, pos_y, pos_z;
    inline static float angle_x = 0.0f, angle_y = 0.0f;

    inline static float angle_x_lim = 30.f;
    inline static float angle_y_lim = 180.f;

    inline static int x_old = 0, y_old = 0;
    inline static int current_scroll = 9; // Approximately 3 cm
    inline static float zoom_per_scroll;

    inline static bool is_holding_mouse = false;
    inline static bool is_updated = false;

    inline static float xy_ratio_limit = 5.0f;
    inline static float speed_constant = 0.f; // The smaller the value, the higher the speed

    inline static int targetMemory_x = 0, targetMemory_y = 0;
    inline static int targetMemoryInit_x = 0, targetMemoryInit_y = 0;

    // For independent motion video
    inline static double video_time_onset = 0.0;
    inline static const double video_time_interval = 42.0; // Roughly 28 FPS

    // XorShift RNG (Use default seeds)
    inline static unsigned long seed[4] = {123456789, 362436069, 521288629, 88675123};
    inline static const int motionRange_x = 3, motionRange_y = 1; // Minimum: 1 (No motion)

    static unsigned long XorShift() {
        unsigned long t = (seed[0] ^ (seed[0] << 11));
        seed[0] = seed[1];
        seed[1] = seed[2];
        seed[2] = seed[3];

        return (seed[3] = (seed[3] ^ (seed[3] >> 19)) ^ (t ^ (t >> 8)));
    }

    // Must be static functions for OpenGL functions
    static void display() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glLoadIdentity();
        glTranslatef(pos_x, pos_y, pos_z);
        glRotatef(angle_x, 1.0f, 0.0f, 0.0f);
        glRotatef(angle_y, 0.0f, 1.0f, 0.0f);

        // During reinforcement
        //if (MaestroFlag) model.draw();
        model.draw();

        glutSwapBuffers();
    }

    static void towardTarget() {

        // angle_y ranges from 0 to 180
        angle_y += (float)(target.x - x_old);
        x_old = target.x;
        if (angle_y > angle_y_lim)
            angle_y = angle_y_lim;
        else if (angle_y < 0.0f)
            angle_y = 0.0f;

        // angle_x ranges from -30 to 30
        angle_x += (float)(target.y - y_old);
        y_old = target.y;
        if (angle_x > angle_x_lim)
            angle_x = angle_x_lim;
        else if (angle_x < -angle_x_lim)
            angle_x = -angle_x_lim;


        // Reference:   https://www.mathsisfun.com/algebra/trig-finding-angle-right-triangle.html
        //              https://lab.syncer.jp/Calculation/Radian-to-Degree/

        // Adjustments
        float adj_y = angle_y * constAdj_y; // Assumes a linear relation. Thus it's not precise but good enough


        // *** Changed (Affects smoothness of motion)
//        float pos_x_diff = abs(target.pos_x - (pos_x + adj_y));
        float pos_x_diff = abs(target.pos_x + awayFloat -  (pos_x + adj_y));

        float pos_y_diff = abs(target.pos_y - pos_y);
        float pos_x_ratio = pos_x_diff/pos_y_diff;
        if (pos_x_ratio > xy_ratio_limit) pos_x_ratio = xy_ratio_limit;

        float dist = sqrt(pos_x_diff * pos_x_diff + pos_y_diff * pos_y_diff);
        float speed = dist / speed_constant;

        // pos_x *** Changed (Affects fish position)
        //if (target.pos_x < pos_x) {
        if (target.pos_x + awayFloat < pos_x) {
            pos_x -= 0.1f * pos_x_ratio * speed;
            glutPostRedisplay();
        }
        //if (target.pos_x > pos_x + adj_y) {
        if (target.pos_x + awayFloat > pos_x + adj_y) {
            pos_x += 0.1f * pos_x_ratio * speed;
            glutPostRedisplay();
        }

        // pos_y (Don't use else if; otherwise, the fish shakes)
        if (target.pos_y < pos_y) {
            pos_y -= 0.1f * speed;
            glutPostRedisplay();
        }
        if (target.pos_y > pos_y) {
            pos_y += 0.1f * speed;
            glutPostRedisplay();
        }
    }

    static void timer(int value) {

        // Load video
        if (initFlag) {
            initFlag = false;

            speed_constant = OpenGLMotionSpeed;

            vcap.open(vfile);
            if(!vcap.isOpened()){
                std::cout << "Error opening video stream or file\n";
            } else {
                vcap >> videoMat; // For avoiding error
            }
            vcap.set(CV_CAP_PROP_POS_MSEC, 1000.0);
        } else {
            if (resetOpenGLVideo) {
                resetOpenGLVideo = false;
                vcap.set(CV_CAP_PROP_POS_MSEC, 1000.0);
            }
        }

        switch (ivMotionSide) {
            case 11:
            case 12:
                if (phaseFlag == 2) {
                    if(vcap.isOpened()){
                        double elapsed = get_dtime() - video_time_onset;
                        if (elapsed > video_time_interval) {
                            video_time_onset = get_dtime();
                            vcap >> videoMat;
                        }
                    }
                }
                break;
            default:
                if(vcap.isOpened()){
                    vcap >> videoMat;
                }
                break;
        }



        // Terminate glutMainLoop()
        if (EndMaestroFlag) glutLeaveMainLoop();

        // During reinforcement
        //if (MaestroFlag) model.draw();
        model.draw();

        switch (phaseFlag) {
            case 1:
                // Nothing
                break;
            case 2:
                switch (ivMotionSide) {
                    case 0: // Right
                    case 1: // Left
                    case 5: // Right
                    case 6: // Left
                    case 7: // Runaway-Right/Chasing-Left
                    case 8: // Chasing-Right/Runaway-Left
                    case 9: // Flipped-Right/Chasing-Left
                    case 10: // Chasing-Right/Flipped-Left
                    case 11:
                    case 12:
                        // While idling
                        if (automaticMotion) {
                            int signX = XorShift() % 2 ? -1 : 1;
                            int signY = XorShift() % 2 ? -1 : 1;
                            int stepX = (XorShift() % motionRange_x) * signX;
                            int stepY = (XorShift() % motionRange_y) * signY;
                            target.x = targetMemory_x + stepX;
                            target.y = targetMemory_y + stepY;
                        }  else {
                            targetMemory_x = target.x;
                            targetMemory_y = target.y;
                        }
                        break;
                }
                break;
            case 3:
//                if (initPhase3Flag) { // Once is enough but just repeats
//                    target.x = targetMemoryInit_x;
//                    target.y = targetMemoryInit_y;
//                }
                break;
        }

//        if (curCallback == callbackID) {
//            adjMotion(extern_target.x, extern_target.y); // Activate when deactivating mouse hover
//            motion(target.x, target.y);
//        } else {
//            // Returns to the center of monitor
//            angle_x = 0.0f;
//            angle_y = (angle_y < 90) ? 0.0f : 180.f;
//            target.x = targetMemoryInit_x;
//            target.y = targetMemoryInit_y;
//            motion(target.x, target.y);
//        }
        switch (callbackID) {
            case 0:
                adjMotion(extern_target0.x, extern_target0.y); // Activate when deactivating mouse hover
                break;
            case 1:
                adjMotion(extern_target1.x, extern_target1.y); // Activate when deactivating mouse hover
                break;
        }

        motion(target.x, target.y);
        towardTarget();

        // Pass OpenGL data to OpenCV Mat
        glutSetWindow(windowID[callbackID]);
        cv::Mat img(HEIGHT, WIDTH, CV_8UC3);
        glReadBuffer(GL_FRONT);
        glReadPixels(0, 0, WIDTH, HEIGHT, GL_BGR_EXT, GL_UNSIGNED_BYTE, img.data);

        switch (ivMotionSide) {
            case 7:
            case 8:
                switch(callbackID) {
                    case 0:
                        cv::flip(img, openGLMat0, 0);
                        break;
                    case 1:
                        cv::flip(img, openGLMat1, 0);
                        break;
                }
                break;
            case 9:
            case 10:
                switch(callbackID) {
                    case 0:
                        cv::flip(img, openGLMat0, 0);
                        break;
                    case 1:
                        openGLMat1 = openGLMat0;
                        break;
                }
                break;
            case 11:
            case 12:
                switch(callbackID) {
                    case 0:
                        cv::flip(img, openGLMat0, 0);
                        break;
                    case 1:
                        openGLMat1 = videoMat;
                        break;
                }
                break;
        }


        // Note: Don't use imshow in a thread. Otherwise, it can crash.

//        if (is_updated) {
//            is_updated = false;
//            glutPostRedisplay();
//        }
//        glutTimerFunc(INTERVAL, timer, 0);
    }

    static void convertCoordinate(int x, int y) {
        target.x = x;
        target.y = y;
        target.pos_x = (float)x * x_ratio + x_min;
        target.pos_y = (float)(HEIGHT - y) * y_ratio + y_min;
    }

    static void motion(int x, int y, int status = 0) {
        float val = 0.f;
        if (motionType == 2) status = 2;
        if (motionType == 3) status = 3;
        if (motionType == 4) status = 4;
        if (motionType == 5) status = 5;
        switch (status) {
            case 0: // Being chased motion
                // Nothing
                break;
            case 1: // Chasing motion
                awayPixel = (target.x > x)? -chaseDist : chaseDist;
                val= (float)(target.x + awayPixel) * x_ratio + x_min;
                awayFloat = val - target.pos_x;

                // angle_y ranges from 0 to 180
                angle_y += (float)(target.x + awayPixel / 25 - x_old);
                x_old = target.x;
                if (angle_y > angle_y_lim)
                    angle_y = angle_y_lim;
                else if (angle_y < 0.0f)
                    angle_y = 0.0f;
                break;
            case 2: // Centerizing motion
                if (x <= idleTarget_x - centMotionRange || x <= -outRange) centMotionDirection = 0; // 0 = Right, 1 = Left
                if (x > idleTarget_x + centMotionRange || x > WIDTH + outRange) centMotionDirection = 1;
                (centMotionDirection)? x -= centMotionSpeed : x += centMotionSpeed;
                break;
            case 3: // Centerizing + WIDTH
            case 4: // Centerizing + WIDTH + No X
                if ((x <= idleTarget_x - centMotionRange && x <= 0) || x <= -outRange) { centMotionDirection = 0;} // 0 = Right, 1 = Left
                if ((x > idleTarget_x + centMotionRange && x > WIDTH) || x > WIDTH + outRange) { centMotionDirection = 1;}
                (centMotionDirection)? x -= centMotionSpeed : x += centMotionSpeed;
                break;
            case 5: // Chasing ver.2
                switch (centMotionDirection) {
                    case 0:
                        centMotionDirection = (x > WIDTH) ? 2 : 0;
                        x += centMotionSpeed;
                        break;
                    case 1:
                        centMotionDirection = (x < 0) ? 3 : 1; // 0 = Right, 1 = Left
                        x -= centMotionSpeed;
                        break;
                }
                break;
            case 6: // Chasing ver.3
                // Nothing
                break;
        }
        convertCoordinate(x, y);
    }

    static void adjMotion(int x, int y) {
        idlingFlag = false;
        automaticMotion = false;

        idleTarget_x = x;
        switch (motionType) {
            case 0:
            case 1:
                if (abs(lastX - x) > motionThresh || abs(lastY - y) > motionThresh) {
                    motion(x, y, motionType);
                }
                break;
            case 2:
            case 3:
                if (abs(lastX - x) > motionThresh || abs(lastY - y) > motionThresh) {
                    centMotionDirection = (x < target.x)? 1 : 0;
                    target.y = y;
                }
                lastX = x;
                lastY = y;
                break;
            case 4:
                if (abs(lastY - y) > motionThresh / 2) target.y = y;
                lastY = y;
                break;
            case 5:
                switch (centMotionDirection) {
                    case 2:
                    case 3:
                        if (abs(target.x - x) < nearX && abs(target.y - y) < nearY) {
                            centMotionDirection = (centMotionDirection == 2)? 1 : 0;
                            if (y < HEIGHT / 2) {
                                target.y = y + (XorShift() % (HEIGHT / 2));
                            } else {
                                target.y = y - (XorShift() % (HEIGHT / 2));
                            }
                        }
                        break;
                }
                break;
            case 6:
                if (abs(target.x - x) < nearX && abs(target.y - y) < nearY) {
                    int hemispNow, hemisNext, halfHeight, halfWidth;
                    halfHeight = HEIGHT / 2;
                    halfWidth = WIDTH / 2;
                    if (target.y < halfHeight) {
                        hemispNow = (target.x < halfWidth)? 0 : 1;
                    } else {
                        hemispNow = (target.x < halfWidth)? 2 : 3;
                    }

                    int hemisCand[2];
                    switch (hemispNow) {
                        case 0:
                        case 2:
                            hemisCand[0] = 1;
                            hemisCand[1] = 3;
                            break;
                        case 1:
                        case 3:
                            hemisCand[0] = 0;
                            hemisCand[1] = 2;
                            break;
                    }
                    hemisNext = hemisCand[XorShift() % 2];

                    int nX, nY;
                    nX = XorShift() % halfWidth;
                    nY = XorShift() % halfHeight;
                    switch (hemisNext) {
                        case 0:
                            target.x = nX / 2;
                            target.y = nY;
                            break;
                        case 1:
                            target.x = halfWidth * 1.2 + nX;
                            target.y = nY;
                            break;
                        case 2:
                            target.x = nX / 2;
                            target.y = halfHeight + nY;
                            break;
                        case 3:
                            target.x = halfWidth * 1.2 + nX;
                            target.y = halfHeight + nY;
                            break;
                    }
                }
                break;
        }
    }

};

class callbacks1 {
public:
    std::string model_name = "../animation_models/mm_frame.obj";

    inline static bool initFlag = true;

    // *** Changed
    // 0 = Being chased (Works), 1 = Chasing ver.1 (Works), 2 = Centerizing motion (Works), 3 = Centerizing + WIDTH (Works),
    // 4 = Centerizing + WIDTH + No X (Works), 5 = Chasing ver.2 (Works), 6 = Chasing ver.3
    inline static int motionType = 0;

    // Common
    inline static const int motionThresh = 16;
    inline static int callbackID = 0;

    // 1: Chasing ver.1
    inline static const int chaseDist = 200; // Should be greater than 180
    inline static int awayPixel = 0;
    inline static float awayFloat = 0.f;

    // 2 & 3 & 4: Centerizing
    inline static int centMotionDirection = 0; // 0 = Right, 1 = Left
    inline static const int centMotionSpeed = 30;//15;
    inline static const int centMotionRange = 350;
    inline static const int outRange = 180;
    inline static int idleTarget_x = 0;
    inline static int lastX = 0, lastY = 0;

    // 5: Chasing ver.2
    inline static const int nearX = 350, nearY = 150;

    inline static measures target;

    // Must be inline for setting variables at the time of initialization
    inline static Model model;

    inline static int POS_X, POS_Y;
    inline static int INTERVAL = 15;

    inline static GLfloat light_pos[] = {-10.0f, 10.0f, 100.00f, 1.0f};

    inline static float pos_x, pos_y, pos_z;
    inline static float angle_x = 0.0f, angle_y = 0.0f;

    inline static float angle_x_lim = 30.f;
    inline static float angle_y_lim = 180.f;

    inline static int x_old = 0, y_old = 0;
    inline static int current_scroll = 9; // Approximately 3 cm
    inline static float zoom_per_scroll;

    inline static bool is_holding_mouse = false;
    inline static bool is_updated = false;

    inline static float xy_ratio_limit = 5.0f;
    inline static float speed_constant = 0.f; // The smaller the value, the higher the speed

    inline static int targetMemory_x = 0, targetMemory_y = 0;
    inline static int targetMemoryInit_x = 0, targetMemoryInit_y = 0;

    // For independent motion video
    inline static double video_time_onset = 0.0;
    inline static const double video_time_interval = 42.0; // Roughly 28 FPS

    // XorShift RNG (Use default seeds)
    inline static unsigned long seed[4] = {123456789, 362436069, 521288629, 88675123};
    inline static const int motionRange_x = 3, motionRange_y = 1; // Minimum: 1 (No motion)

    static unsigned long XorShift() {
        unsigned long t = (seed[0] ^ (seed[0] << 11));
        seed[0] = seed[1];
        seed[1] = seed[2];
        seed[2] = seed[3];

        return (seed[3] = (seed[3] ^ (seed[3] >> 19)) ^ (t ^ (t >> 8)));
    }

    // Must be static functions for OpenGL functions
    static void display() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glLoadIdentity();
        glTranslatef(pos_x, pos_y, pos_z);
        glRotatef(angle_x, 1.0f, 0.0f, 0.0f);
        glRotatef(angle_y, 0.0f, 1.0f, 0.0f);

        // During reinforcement
        //if (MaestroFlag) model.draw();
        model.draw();

        glutSwapBuffers();
    }

    static void towardTarget() {

        // angle_y ranges from 0 to 180
        angle_y += (float)(target.x - x_old);
        x_old = target.x;
        if (angle_y > angle_y_lim)
            angle_y = angle_y_lim;
        else if (angle_y < 0.0f)
            angle_y = 0.0f;

        // angle_x ranges from -30 to 30
        angle_x += (float)(target.y - y_old);
        y_old = target.y;
        if (angle_x > angle_x_lim)
            angle_x = angle_x_lim;
        else if (angle_x < -angle_x_lim)
            angle_x = -angle_x_lim;


        // Reference:   https://www.mathsisfun.com/algebra/trig-finding-angle-right-triangle.html
        //              https://lab.syncer.jp/Calculation/Radian-to-Degree/

        // Adjustments
        float adj_y = angle_y * constAdj_y; // Assumes a linear relation. Thus it's not precise but good enough


        // *** Changed (Affects smoothness of motion)
//        float pos_x_diff = abs(target.pos_x - (pos_x + adj_y));
        float pos_x_diff = abs(target.pos_x + awayFloat -  (pos_x + adj_y));

        float pos_y_diff = abs(target.pos_y - pos_y);
        float pos_x_ratio = pos_x_diff/pos_y_diff;
        if (pos_x_ratio > xy_ratio_limit) pos_x_ratio = xy_ratio_limit;

        float dist = sqrt(pos_x_diff * pos_x_diff + pos_y_diff * pos_y_diff);
        float speed = dist / speed_constant;

        // pos_x *** Changed (Affects fish position)
        //if (target.pos_x < pos_x) {
        if (target.pos_x + awayFloat < pos_x) {
            pos_x -= 0.1f * pos_x_ratio * speed;
            glutPostRedisplay();
        }
        //if (target.pos_x > pos_x + adj_y) {
        if (target.pos_x + awayFloat > pos_x + adj_y) {
            pos_x += 0.1f * pos_x_ratio * speed;
            glutPostRedisplay();
        }

        // pos_y (Don't use else if; otherwise, the fish shakes)
        if (target.pos_y < pos_y) {
            pos_y -= 0.1f * speed;
            glutPostRedisplay();
        }
        if (target.pos_y > pos_y) {
            pos_y += 0.1f * speed;
            glutPostRedisplay();
        }
    }

    static void timer(int value) {

        // Load video
        if (initFlag) {
            initFlag = false;

            speed_constant = OpenGLMotionSpeed;

            vcap.open(vfile);
            if(!vcap.isOpened()){
                std::cout << "Error opening video stream or file\n";
            } else {
                vcap >> videoMat; // For avoiding error
            }
            vcap.set(CV_CAP_PROP_POS_MSEC, 1000.0);
        } else {
            if (resetOpenGLVideo) {
                resetOpenGLVideo = false;
                vcap.set(CV_CAP_PROP_POS_MSEC, 1000.0);
            }
        }

        switch (ivMotionSide) {
            case 11:
            case 12:
                if (phaseFlag == 2) {
                    if(vcap.isOpened()){
                        double elapsed = get_dtime() - video_time_onset;
                        if (elapsed > video_time_interval) {
                            video_time_onset = get_dtime();
                            vcap >> videoMat;
                        }
                    }
                }
                break;
            default:
                if(vcap.isOpened()){
                    vcap >> videoMat;
                }
                break;
        }



        // Terminate glutMainLoop()
        if (EndMaestroFlag) glutLeaveMainLoop();

        // During reinforcement
        //if (MaestroFlag) model.draw();
        model.draw();

        switch (phaseFlag) {
            case 1:
                // Nothing
                break;
            case 2:
                switch (ivMotionSide) {
                    case 0: // Right
                    case 1: // Left
                    case 5: // Right
                    case 6: // Left
                    case 7: // Runaway-Right/Chasing-Left
                    case 8: // Chasing-Right/Runaway-Left
                    case 9: // Flipped-Right/Chasing-Left
                    case 10: // Chasing-Right/Flipped-Left
                    case 11:
                    case 12:
                        // While idling
                        if (automaticMotion) {
                            int signX = XorShift() % 2 ? -1 : 1;
                            int signY = XorShift() % 2 ? -1 : 1;
                            int stepX = (XorShift() % motionRange_x) * signX;
                            int stepY = (XorShift() % motionRange_y) * signY;
                            target.x = targetMemory_x + stepX;
                            target.y = targetMemory_y + stepY;
                        }  else {
                            targetMemory_x = target.x;
                            targetMemory_y = target.y;
                        }
                        break;
                }
                break;
            case 3:
//                if (initPhase3Flag) { // Once is enough but just repeats
//                    target.x = targetMemoryInit_x;
//                    target.y = targetMemoryInit_y;
//                }
                break;
        }

//        if (curCallback == callbackID) {
//            adjMotion(extern_target.x, extern_target.y); // Activate when deactivating mouse hover
//            motion(target.x, target.y);
//        } else {
//            // Returns to the center of monitor
//            angle_x = 0.0f;
//            angle_y = (angle_y < 90) ? 0.0f : 180.f;
//            target.x = targetMemoryInit_x;
//            target.y = targetMemoryInit_y;
//            motion(target.x, target.y);
//        }
        switch (callbackID) {
            case 0:
                adjMotion(extern_target0.x, extern_target0.y); // Activate when deactivating mouse hover
                break;
            case 1:
                adjMotion(extern_target1.x, extern_target1.y); // Activate when deactivating mouse hover
                break;
        }

        motion(target.x, target.y);
        towardTarget();

        // Pass OpenGL data to OpenCV Mat
        glutSetWindow(windowID[callbackID]);
        cv::Mat img(HEIGHT, WIDTH, CV_8UC3);
        glReadBuffer(GL_FRONT);
        glReadPixels(0, 0, WIDTH, HEIGHT, GL_BGR_EXT, GL_UNSIGNED_BYTE, img.data);

        switch (ivMotionSide) {
            case 7:
            case 8:
                switch(callbackID) {
                    case 0:
                        cv::flip(img, openGLMat0, 0);
                        break;
                    case 1:
                        cv::flip(img, openGLMat1, 0);
                        break;
                }
                break;
            case 9:
            case 10:
                switch(callbackID) {
                    case 0:
                        cv::flip(img, openGLMat0, 0);
                        break;
                    case 1:
                        openGLMat1 = openGLMat0;
                        break;
                }
                break;
            case 11:
            case 12:
                switch(callbackID) {
                    case 0:
                        cv::flip(img, openGLMat0, 0);
                        break;
                    case 1:
                        openGLMat1 = videoMat;
                        break;
                }
                break;
        }


        // Note: Don't use imshow in a thread. Otherwise, it can crash.

//        if (is_updated) {
//            is_updated = false;
//            glutPostRedisplay();
//        }
//        glutTimerFunc(INTERVAL, timer, 0);
    }

    static void convertCoordinate(int x, int y) {
        target.x = x;
        target.y = y;
        target.pos_x = (float)x * x_ratio + x_min;
        target.pos_y = (float)(HEIGHT - y) * y_ratio + y_min;
    }

    static void motion(int x, int y, int status = 0) {
        float val = 0.f;
        if (motionType == 2) status = 2;
        if (motionType == 3) status = 3;
        if (motionType == 4) status = 4;
        if (motionType == 5) status = 5;
        switch (status) {
            case 0: // Being chased motion
                // Nothing
                break;
            case 1: // Chasing motion
                awayPixel = (target.x > x)? -chaseDist : chaseDist;
                val= (float)(target.x + awayPixel) * x_ratio + x_min;
                awayFloat = val - target.pos_x;

                // angle_y ranges from 0 to 180
                angle_y += (float)(target.x + awayPixel / 25 - x_old);
                x_old = target.x;
                if (angle_y > angle_y_lim)
                    angle_y = angle_y_lim;
                else if (angle_y < 0.0f)
                    angle_y = 0.0f;
                break;
            case 2: // Centerizing motion
                if (x <= idleTarget_x - centMotionRange || x <= -outRange) centMotionDirection = 0; // 0 = Right, 1 = Left
                if (x > idleTarget_x + centMotionRange || x > WIDTH + outRange) centMotionDirection = 1;
                (centMotionDirection)? x -= centMotionSpeed : x += centMotionSpeed;
                break;
            case 3: // Centerizing + WIDTH
            case 4: // Centerizing + WIDTH + No X
                if ((x <= idleTarget_x - centMotionRange && x <= 0) || x <= -outRange) { centMotionDirection = 0;} // 0 = Right, 1 = Left
                if ((x > idleTarget_x + centMotionRange && x > WIDTH) || x > WIDTH + outRange) { centMotionDirection = 1;}
                (centMotionDirection)? x -= centMotionSpeed : x += centMotionSpeed;
                break;
            case 5: // Chasing ver.2
                switch (centMotionDirection) {
                    case 0:
                        centMotionDirection = (x > WIDTH) ? 2 : 0;
                        x += centMotionSpeed;
                        break;
                    case 1:
                        centMotionDirection = (x < 0) ? 3 : 1; // 0 = Right, 1 = Left
                        x -= centMotionSpeed;
                        break;
                }
                break;
            case 6: // Chasing ver.3
                // Nothing
                break;
        }
        convertCoordinate(x, y);
    }

    static void adjMotion(int x, int y) {
        idlingFlag = false;
        automaticMotion = false;

        idleTarget_x = x;
        switch (motionType) {
            case 0:
            case 1:
                if (abs(lastX - x) > motionThresh || abs(lastY - y) > motionThresh) {
                    motion(x, y, motionType);
                }
                break;
            case 2:
            case 3:
                if (abs(lastX - x) > motionThresh || abs(lastY - y) > motionThresh) {
                    centMotionDirection = (x < target.x)? 1 : 0;
                    target.y = y;
                }
                lastX = x;
                lastY = y;
                break;
            case 4:
                if (abs(lastY - y) > motionThresh / 2) target.y = y;
                lastY = y;
                break;
            case 5:
                switch (centMotionDirection) {
                    case 2:
                    case 3:
                        if (abs(target.x - x) < nearX && abs(target.y - y) < nearY) {
                            centMotionDirection = (centMotionDirection == 2)? 1 : 0;
                            if (y < HEIGHT / 2) {
                                target.y = y + (XorShift() % (HEIGHT / 2));
                            } else {
                                target.y = y - (XorShift() % (HEIGHT / 2));
                            }
                        }
                        break;
                }
                break;
            case 6:
                if (abs(target.x - x) < nearX && abs(target.y - y) < nearY) {
                    int hemispNow, hemisNext, halfHeight, halfWidth;
                    halfHeight = HEIGHT / 2;
                    halfWidth = WIDTH / 2;
                    if (target.y < halfHeight) {
                        hemispNow = (target.x < halfWidth)? 0 : 1;
                    } else {
                        hemispNow = (target.x < halfWidth)? 2 : 3;
                    }

                    int hemisCand[2];
                    switch (hemispNow) {
                        case 0:
                        case 2:
                            hemisCand[0] = 1;
                            hemisCand[1] = 3;
                            break;
                        case 1:
                        case 3:
                            hemisCand[0] = 0;
                            hemisCand[1] = 2;
                            break;
                    }
                    hemisNext = hemisCand[XorShift() % 2];

                    int nX, nY;
                    nX = XorShift() % halfWidth;
                    nY = XorShift() % halfHeight;
                    switch (hemisNext) {
                        case 0:
                            target.x = nX / 2;
                            target.y = nY;
                            break;
                        case 1:
                            target.x = halfWidth * 1.2 + nX;
                            target.y = nY;
                            break;
                        case 2:
                            target.x = nX / 2;
                            target.y = halfHeight + nY;
                            break;
                        case 3:
                            target.x = halfWidth * 1.2 + nX;
                            target.y = halfHeight + nY;
                            break;
                    }
                }
                break;
        }
    }

};


class modelApp {
private:
    inline static callbacks0 cb0;
    inline static callbacks1 cb1;
    static const int INTERVAL = 0;

    static void mainTimer(int value) {
        glutSetWindow(windowID[0]);
        glutTimerFunc(0, cb0.timer, 0);
        glutPostRedisplay();

        glutSetWindow(windowID[1]);
        glutTimerFunc(0, cb1.timer, 0);
        glutPostRedisplay();

        glutTimerFunc(INTERVAL, mainTimer, 0);
    }

    void init1(int ID, int motion_type) {
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        glLightfv(GL_LIGHT0, GL_POSITION, cb1.light_pos);
        //glClearColor(0.4f, 0.4f, 0.4f, 1.0f);
        glClearColor(1.f, 1.f, 1.f, 1.0f); // tk replaced

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        // Replaced (2021-11-8)
        // gluPerspective(20.0, 1.0, 1.0, 2000.0);
        // See https://atelier-yoka.com/dev_android/p_main.php?file=apiglugluperspective
        gluPerspective(20.0, (GLdouble)WIDTH / (GLdouble)HEIGHT, 1.0, 2000.0);

        glMatrixMode(GL_MODELVIEW);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_LINE_SMOOTH);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glEnable(GL_TEXTURE_2D);
        glEnable(GL_DEPTH_TEST);

        cb1.model.load(cb1.model_name.c_str());

        // pos_x, pos_y, and pos_z represent locations on the window
        cb1.pos_x = cb1.model.pos_x;
        cb1.pos_y = cb1.model.pos_y;
        cb1.pos_z = cb1.model.pos_z - 1.0f;

        cb1.zoom_per_scroll = -cb1.model.pos_z / 10.0f;

        // Initialize the size
        cb1.pos_z -= cb1.zoom_per_scroll * cb1.current_scroll;

//        extern_target1.x = WIDTH / 2;
//        extern_target1.y = HEIGHT / 2;
//
//        cb1.target.x = WIDTH / 2;
//        cb1.target.y = HEIGHT / 2;
//
//        cb1.targetMemory_x = extern_target1.x;
//        cb1.targetMemory_y = extern_target1.y;
//        cb1.targetMemoryInit_x = extern_target1.x;
//        cb1.targetMemoryInit_y = extern_target1.y;


        cb1.target.x = WIDTH / 2;
        cb1.target.y = HEIGHT / 2;

        cb1.targetMemory_x = cb1.target.x;
        cb1.targetMemory_y = cb1.target.y;
        cb1.targetMemoryInit_x = cb1.target.x;
        cb1.targetMemoryInit_y = cb1.target.y;

        cb1.motionType = motion_type;
        cb1.callbackID = ID;

    }

    void init0(int ID, int motion_type) {
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        glLightfv(GL_LIGHT0, GL_POSITION, cb0.light_pos);
        //glClearColor(0.4f, 0.4f, 0.4f, 1.0f);
        glClearColor(1.f, 1.f, 1.f, 1.0f); // tk replaced

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        // Replaced (2021-11-8)
        // gluPerspective(20.0, 1.0, 1.0, 2000.0);
        // See https://atelier-yoka.com/dev_android/p_main.php?file=apiglugluperspective
        gluPerspective(20.0, (GLdouble)WIDTH / (GLdouble)HEIGHT, 1.0, 2000.0);

        glMatrixMode(GL_MODELVIEW);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_LINE_SMOOTH);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glEnable(GL_TEXTURE_2D);
        glEnable(GL_DEPTH_TEST);

        cb0.model.load(cb0.model_name.c_str());

        // pos_x, pos_y, and pos_z represent locations on the window
        cb0.pos_x = cb0.model.pos_x;
        cb0.pos_y = cb0.model.pos_y;
        cb0.pos_z = cb0.model.pos_z - 1.0f;

        cb0.zoom_per_scroll = -cb0.model.pos_z / 10.0f;

        // Initialize the size
        cb0.pos_z -= cb0.zoom_per_scroll * cb0.current_scroll;

//        extern_target0.x = WIDTH / 2;
//        extern_target0.y = HEIGHT / 2;
//
//        cb0.target.x = WIDTH / 2;
//        cb0.target.y = HEIGHT / 2;
//
//        cb0.targetMemory_x = extern_target0.x;
//        cb0.targetMemory_y = extern_target0.y;
//        cb0.targetMemoryInit_x = extern_target0.x;
//        cb0.targetMemoryInit_y = extern_target0.y;

        cb0.target.x = WIDTH / 2;
        cb0.target.y = HEIGHT / 2;

        cb0.targetMemory_x = cb0.target.x;
        cb0.targetMemory_y = cb0.target.y;
        cb0.targetMemoryInit_x = cb0.target.x;
        cb0.targetMemoryInit_y = cb0.target.y;

        cb0.motionType = motion_type;
        cb0.callbackID = ID;

    }

public:
    void run(int argc, char **argv) {

        // Set motion type
        // 0 = Being chased (Works), 1 = Chasing ver.1 (Works), 2 = Centerizing motion (Works), 3 = Centerizing + WIDTH (Works),
        // 4 = Centerizing + WIDTH + No X (Works), 5 = Chasing ver.2 (Works), 6 = Chasing ver.3
        int win[2];
        win[0] = 0;
        win[1] = 0;
        switch (ivMotionSide) {
            case 7:
            case 8:
                win[0] = 0;
                win[1] = 6;
                break;
        }

        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE);
        glEnable(GL_MULTISAMPLE);
        glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_NICEST);
        glutSetOption(GLUT_MULTISAMPLE, 8);

        int ID = 0;

        cb0.POS_X = (1280 - WIDTH) >> 1;
        cb0.POS_Y = (720 - HEIGHT) >> 1;

        glutInitWindowPosition(cb0.POS_X, cb0.POS_Y);
        glutInitWindowSize(WIDTH, HEIGHT);
        windowID[ID] = glutCreateWindow("OpenGL 0");

        // Add more options and then load a .obj model
        init0(ID, win[ID]);

        glutDisplayFunc(cb0.display);

        // Prepare for leaving from glutMainLoop();
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_GLUTMAINLOOP_RETURNS);



        /* Second Window */
        ID = 1;
        cb1.POS_X = (1280 - WIDTH) >> 1;
        cb1.POS_Y = (720 - HEIGHT) >> 1;
        glutInitWindowPosition(cb1.POS_X, cb1.POS_Y);
        glutInitWindowSize(WIDTH, HEIGHT);
        windowID[ID] = glutCreateWindow("OpenGL 1");
        init1(ID, win[ID]);
        glutDisplayFunc(cb1.display);
        //glutPassiveMotionFunc(cb1.adjMotion);
        //glutTimerFunc(0, cb1.timer, 0);
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_GLUTMAINLOOP_RETURNS);
        glutTimerFunc(0, mainTimer, 0);

        glutMainLoop();
    }
};

#endif