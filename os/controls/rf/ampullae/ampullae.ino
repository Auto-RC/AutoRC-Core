#include <Wire.h>

float THR_MAX = 1925;
float THR_MIN = 1000;
float STR_MAX = 1990;
float STR_MIN = 1050;
float SWB_MAX = 1900;
float SWB_MIN = 1000;
float SWC_MAX = 1000;
float SWC_MIN = 1000;
int THR = 1;
int STR = 2;
int SWB = 3;
int SWC = 4;

// These hold the input from the receiver
int thr_input;
int str_input;
int swb_input;
int swc_input;

float thr;
float str;
float swb;
float swc;

int thr_output;
int str_output;
int swb_output;
int swc_output;

void setup()
{
    Serial.begin(9600);

    pinMode(2,INPUT);
    pinMode(4,INPUT);
    pinMode(6,INPUT);
    pinMode(7,INPUT);
//    Wire.begin(8);                // join i2c bus with address #8
//    Wire.onRequest(requestEvent); // register event
}

void loop()
{
    thr_input = pulseIn(4,HIGH,1000000);
    str_input = pulseIn(2,HIGH,1000000);
    swb_input = pulseIn(6,HIGH,1000000);
    swc_input = pulseIn(7,HIGH,1000000);

    thr = thr_input - THR_MIN;
    str = str_input - STR_MIN;
    swb = swb_input - SWB_MIN;
    swc = swc_input - SWC_MIN;

    thr_output = ( thr/(THR_MAX-THR_MIN) )*255;
    str_output = ( str/(STR_MAX-STR_MIN) )*255;
    swb_output = ( swb/(SWB_MAX-SWB_MIN) )*255;
    swc_output = ( swc/(SWC_MAX-SWC_MIN) )*255;

    Serial.print(thr_output);
    Serial.print(" ");
    Serial.print(str_output);
    Serial.print(" ");
    Serial.print(swb_output);
    Serial.print(" ");
    Serial.print(swc_output);
    Serial.print(" ");

    delay(10);
}

//void encode_signal(int type, int value)
//{
//}
//
//// function that executes whenever data is requested by master
//// this function is registered as an event, see setup()
//void requestEvent() {
//  Wire.write("hello "); // respond with message of 6 bytes
//  // as expected by master
//}
