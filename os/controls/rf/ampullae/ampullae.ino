#include <Wire.h>

// Receiver calibration constants
float THR_MAX = 1990;
float THR_MIN = 990;
float STR_MAX = 1990;
float STR_MIN = 990;
float SWB_MAX = 1990;
float SWB_MIN = 990;
float SWC_MAX = 1988;
float SWC_MIN = 990;

// I2C Format
byte THR = 0;
byte STR = 1;
byte SWB = 2;
byte SWC = 3;
int I2C_MAX = 64;



// These hold the input from the receiver
int thr_input;
int str_input;
int swb_input;
int swc_input;

// These hold the intermediate values
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

    thr_output = ( (thr_input - THR_MIN)/(THR_MAX-THR_MIN) )*I2C_MAX;
    str_output = ( (str_input - STR_MIN)/(STR_MAX-STR_MIN) )*I2C_MAX;
    swb_output = ( (swb_input -SWB_MIN)/(SWB_MAX-SWB_MIN) )*I2C_MAX;
    swc_output = ( (swc_input -SWC_MIN)/(SWC_MAX-SWC_MIN) )*I2C_MAX;

    Serial.print(thr_output);
    Serial.print(" ");
    Serial.print(thr_input);
    Serial.print(" ");
    Serial.print(str_output);
    Serial.print(" ");
    Serial.print(str_input);
    Serial.print(" ");
//    Serial.print(swb_output);
//    Serial.print(" ");
//    Serial.print(swb_input);
//    Serial.print(" ");
//    Serial.print(swc_output);
//    Serial.print(" ");
//    Serial.print(swc_input);
//    Serial.println(" ");

   
    encode_signal(STR, str_output);
    
    delay(10);
}

void encode_signal(int type, byte value)
{ 
    byte type_bits = type << 6;
    byte val_bits = value;
    byte send_bits = type_bits + val_bits;

    Serial.print(type_bits);
    Serial.print(" ");
    Serial.print(val_bits);
    Serial.print(" ");
    Serial.println(send_bits);
}
//
//// function that executes whenever data is requested by master
//// this function is registered as an event, see setup()
//void requestEvent() {
//  Wire.write("hello "); // respond with message of 6 bytes
//  // as expected by master
//}
