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

// The values to send to the car computer
int thr_output;
int str_output;
int swb_output;
int swc_output;

// The bytes which are actually sent to the car computer
unsigned char thr = 0;
unsigned char str = 0;
unsigned char swb = 0;
unsigned char swc = 0;

void setup()
{
    Serial.begin(9600);

    pinMode(2,INPUT);
    pinMode(4,INPUT);
    pinMode(6,INPUT);
    pinMode(7,INPUT);
    
    Wire.begin(0x05);              // join i2c bus with address #5
    Wire.onRequest(send_controls); // register event
}

void loop()
{
    read_rf();    
    
    encode_signal(THR, thr_output);
    encode_signal(STR, str_output);
    encode_signal(SWB, swb_output);
    encode_signal(SWC, swc_output);

    delay(10);
}

void send_controls()
{
    Wire.write(thr);
    Wire.write(str);
    Wire.write(swb);
    Wire.write(swc);
}

void read_rf()
{
    thr_input = pulseIn(4,HIGH,1000000);
    str_input = pulseIn(2,HIGH,1000000);
    swb_input = pulseIn(6,HIGH,1000000);
    swc_input = pulseIn(7,HIGH,1000000);

    thr_output = ( (thr_input - THR_MIN)/(THR_MAX-THR_MIN) )*I2C_MAX;
    str_output = ( (str_input - STR_MIN)/(STR_MAX-STR_MIN) )*I2C_MAX;
    swb_output = ( (swb_input -SWB_MIN)/(SWB_MAX-SWB_MIN) )*I2C_MAX;
    swc_output = ( (swc_input -SWC_MIN)/(SWC_MAX-SWC_MIN) )*I2C_MAX;

    // DEBUG PRINTS
    // -------------------------------------
    Serial.print("Thr: ");
    Serial.print(thr);
    Serial.print(" Str: ");
    Serial.print(str);
    Serial.print(" SWB: ");
    Serial.print(swb);
    Serial.print(" SWC: ");
    Serial.println(swc);
    // -------------------------------------
    
    delay(10);
}

void encode_signal(int type, int value)
{   
    delay(10);
    byte type_bits = type << 6;
    byte val_bits = value;
    byte send_bits = type_bits + val_bits;

    if (type == SWC)
    {
      if (send_bits == 0)
      {
        send_bits = send_bits - 1;
      }
    }

    if (type == THR)
    {
      thr = send_bits;
    }
    else if (type == STR)
    {
      str = send_bits;
    }
    else if (type == SWB)
    {
      swb = send_bits; 
    }
    else if (type == SWC)
    {
      swc = send_bits; 
    }
}
