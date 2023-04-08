#ifndef _MAESTRO_H_
#define _MAESTRO_H_

#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>  // for gettimeofday()
#include "extern.h"


#define BAUDRATE B9600  // Default value of Arduino
#define BUFFSIZE 255  // Default baudrate of Arduino

static int fd = -1;
char buffer[BUFFSIZE] = {0};

class MaestroApp {
private:
    const char* SERIAL_PORT;
    struct termios oldtio, newtio;

    // Maestro parameters
    unsigned char Channel_Feeder = 0;
    unsigned char Channel_Vibrator = 0;
    unsigned char Channel_FeederLight = 0;
    unsigned char Channel_Houselight = 0;
    unsigned char Channel_Fan = 0;
    unsigned char FEEDER_ON[4] = {};
    unsigned char FEEDER_OFF[4] = {};
    unsigned char VIBRATOR_ON[4] = {};
    unsigned char VIBRATOR_OFF[4] = {};
    unsigned char FEEDERLIGHT_ON[4] = {};
    unsigned char FEEDERLIGHT_OFF[4] = {};
    unsigned char HOUSELIGHT_ON[4] = {};
    unsigned char HOUSELIGHT_OFF[4] = {};
    unsigned char FAN_ON[4] = {};
    unsigned char FAN_OFF[4] = {};

public:
    MaestroApp();
    ~MaestroApp();
    void run();
    void turn_on_HL();
    void turn_off_HL();

private:
    int maestroSetTarget(const unsigned char* command);
    void PulseInterval(double PulseWidth);
    void Reinforcement();
    void SetMaestroParameters();
    int OpenSerialPort();
};

#endif
