#include "maestro.h"

void MaestroApp::turn_on_HL(){
    maestroSetTarget(HOUSELIGHT_ON);
}

void MaestroApp::turn_off_HL(){
    maestroSetTarget(HOUSELIGHT_OFF);
}

void MaestroApp::run() {
    try {
        maestroSetTarget(HOUSELIGHT_ON);

        while (1) {
            if (MaestroFlag) {
                Reinforcement();
                SRarranged = false;
                MaestroFlag = false;
            }

            usleep(100000); // 0.1 sec
            if (EndMaestroFlag) {
                maestroSetTarget(HOUSELIGHT_OFF);
                break;
            }
        }
    }
    catch (std::exception& ex) {
        std::cout << ex.what() << std::endl;
        maestroSetTarget(HOUSELIGHT_OFF);
        return;
    }
}

int MaestroApp::maestroSetTarget(const unsigned char* command) {
    int byte_count = 0;
    while (byte_count < 1) {
        byte_count = write(fd, command, 4); // Send 4 bytes to Maestro (Ref: https://www.sejuku.net/blog/24793)
        if (byte_count == -1) {
            perror("Error SignalToMaestro");
            return -1;
        }
    }
    return 0;
}

void MaestroApp::PulseInterval(double PulseWidth) {

    double st, now;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    st = (double)tv.tv_sec + (double)tv.tv_usec*1.0e-06;

    while (1) {
        gettimeofday(&tv, NULL);
        now = (double)tv.tv_sec + (double)tv.tv_usec*1.0e-06;
        if (now - st >= PulseWidth) break;
    }
}

void MaestroApp::Reinforcement() {
    clock_t FeederOnsetTime = clock();

    // Feeder & Feeder light on
    maestroSetTarget(FEEDERLIGHT_ON);
    maestroSetTarget(FEEDER_ON);

    // Vibration
    maestroSetTarget(VIBRATOR_ON);
    PulseInterval(0.06);
    maestroSetTarget(VIBRATOR_OFF);
    PulseInterval(0.03);
    maestroSetTarget(VIBRATOR_ON);
    PulseInterval(0.08);
    maestroSetTarget(VIBRATOR_OFF);
    PulseInterval(0.03);
    maestroSetTarget(VIBRATOR_ON);
    PulseInterval(0.08);
    maestroSetTarget(VIBRATOR_OFF);
    PulseInterval(0.03);
    maestroSetTarget(VIBRATOR_ON);
    PulseInterval(0.06);

    maestroSetTarget(FEEDER_OFF);
    PulseInterval(0.01);
    maestroSetTarget(VIBRATOR_OFF);

    for (int i = 0; i < 5; ++i) {
        maestroSetTarget(FEEDERLIGHT_OFF);
        PulseInterval(0.2);
        maestroSetTarget(FEEDERLIGHT_ON);
        PulseInterval(0.2);
    }

    maestroSetTarget(FEEDERLIGHT_OFF);

    while ((clock() - FeederOnsetTime) < SRduration) PulseInterval(0.1);
}

void MaestroApp::SetMaestroParameters() {
    Channel_Feeder = 0;
    Channel_Vibrator = 1;
    Channel_FeederLight = 2;
    Channel_Houselight = 3;
    Channel_Fan = 4;

    if (strcmp(SERIAL_PORT, "/dev/ttyACM0") == 0) {
        FEEDER_ON[0] = 0x84;
        FEEDER_ON[1] = Channel_Feeder;
        FEEDER_ON[2] = 0x30;// Set on 2021-4-29
        FEEDER_ON[3] = 0x36;
    }
    else if (strcmp(SERIAL_PORT, "/dev/ttyACM2") == 0) {
        FEEDER_ON[0] = 0x84;
        FEEDER_ON[1] = Channel_Feeder;
        FEEDER_ON[2] = 0x0;//0x10;// Set on 2021-4-29, to 0x0 on 2021-5-22
        FEEDER_ON[3] = 0x36;
    }

    FEEDER_OFF[0] = 0x84;
    FEEDER_OFF[1] = Channel_Feeder;
    FEEDER_OFF[2] = 0x7F;
    FEEDER_OFF[3] = 0x7F;

    VIBRATOR_ON[0] = 0x84;
    VIBRATOR_ON[1] = Channel_Vibrator;
    VIBRATOR_ON[2] = 0x0;
    VIBRATOR_ON[3] = 0x3E;

    VIBRATOR_OFF[0] = 0x84;
    VIBRATOR_OFF[1] = Channel_Vibrator;
    VIBRATOR_OFF[2] = 0x1;
    VIBRATOR_OFF[3] = 0x0;

    FEEDERLIGHT_ON[0] = 0x84;
    FEEDERLIGHT_ON[1] = Channel_FeederLight;
    FEEDERLIGHT_ON[2] = 0x70;
    FEEDERLIGHT_ON[3] = 0x2E;

    FEEDERLIGHT_OFF[0] = 0x84;
    FEEDERLIGHT_OFF[1] = Channel_FeederLight;
    FEEDERLIGHT_OFF[2] = 0x1;
    FEEDERLIGHT_OFF[3] = 0x0;

    HOUSELIGHT_ON[0] = 0x84;
    HOUSELIGHT_ON[1] = Channel_Houselight;
    HOUSELIGHT_ON[2] = 0x0;
    HOUSELIGHT_ON[3] = 0x3E;

    HOUSELIGHT_OFF[0] = 0x84;
    HOUSELIGHT_OFF[1] = Channel_Houselight;
    HOUSELIGHT_OFF[2] = 0x1;
    HOUSELIGHT_OFF[3] = 0x0;

    FAN_ON[0] = 0x84;
    FAN_ON[1] = Channel_Fan;
    FAN_ON[2] = 0x70;
    FAN_ON[3] = 0x2E;

    FAN_OFF[0] = 0x84;
    FAN_OFF[1] = Channel_Fan;
    FAN_OFF[2] = 0x1;
    FAN_OFF[3] = 0x0;
}

int MaestroApp::OpenSerialPort() {

    /* Open a serial interface
    *   fd (file descriptor) is the name of serial interface
    *   SERIAL_PORT:  pathname
    *   O_RDWR: Option for read and write
    *   O_NOCTTY: This is a device file. So don't assign a controlling terminal (which controls the process, for example, with Ctrl-C) to this serial port.
    */
    fd = open(SERIAL_PORT, O_RDWR | O_NOCTTY);
    if (fd < 0) {
        perror(SERIAL_PORT);
        return -1;
    }

    /* Interface for controlling a serial port
     *   tcgetattr...: Save the current setting of fd (serial port) to oldtio
     *   memset...: Clear (set 0) the setting of a new serial port
     */

    tcgetattr(fd, &oldtio);
    memset(&newtio, 0, sizeof(newtio));

    /* Setting for input (c_iflag)
     *   IGNPAR: Ignore data if there is a parity error or a framing error
     *   ICRNL:  Convert carriage return (CR; return to the beginning of a sentence) to new line (NL; "\n")
     */
    newtio.c_iflag = IGNPAR | ICRNL;

    /* Setting for controlling hardware (c_cflag)
     *   BAUDRATE: Set baudrate
     *   CS8 : 8n1 (8 bit, Non-parity, stop bit is 1)
     *   CLOCAL: Local connection (Ignore modem control)
     *   CREAD: Enable receiving characters
     */
    newtio.c_cflag = BAUDRATE | CS8 | CLOCAL | CREAD;

    /* Setting for local modes (c_lflag)
     *   0 = Non-cannonical mode
     */
    newtio.c_lflag = 0;

    /* Setting for special characters (c_cc)
     *   VTIME = 0: Disable the between-character timer
     */
    newtio.c_cc[VTIME] = 0;

    tcsetattr(fd, TCSANOW, &newtio); // Make the new setting effective right now

    return 0;
}

MaestroApp::MaestroApp() {
    SERIAL_PORT = COM.c_str();
    SetMaestroParameters();
    OpenSerialPort();
}

MaestroApp::~MaestroApp() {
    tcsetattr(fd, TCSANOW, &oldtio); // Change back to the original setting of fd
    close(fd); // Close the fd (serial port)
}
