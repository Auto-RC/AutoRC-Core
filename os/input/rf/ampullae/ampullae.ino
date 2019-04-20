// Receiver calibration constants
float THR_MAX = 1990;
float THR_MIN = 990;
float STR_MAX = 1990;
float STR_MIN = 990;
float SWB_MAX = 1990;
float SWB_MIN = 990;
float SWC_MAX = 1988;
float SWC_MIN = 990;
int MAX = 89;

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

void setup()
{
    Serial.begin(9600);

    pinMode(2,INPUT);
    pinMode(4,INPUT);
    pinMode(6,INPUT);
    pinMode(7,INPUT);
}

void loop()
{
    read_rf();    

    Serial.print(thr_output);
    Serial.print(str_output);
    Serial.print(swc_output);
    Serial.print(swb_output);
        
    delay(20);
}

void read_rf()
{
    thr_input = pulseIn(4,HIGH,1000000);
    str_input = pulseIn(2,HIGH,1000000);
    swb_input = pulseIn(6,HIGH,1000000);
    swc_input = pulseIn(7,HIGH,1000000);

    thr_output = ( (thr_input - THR_MIN)/(THR_MAX-THR_MIN) )*MAX + 10;
    str_output = ( (str_input - STR_MIN)/(STR_MAX-STR_MIN) )*MAX + 10;
    swb_output = ( (swb_input - SWB_MIN)/(SWB_MAX-SWB_MIN) )*MAX + 10;
    swc_output = ( (swc_input - SWC_MIN)/(SWC_MAX-SWC_MIN) )*MAX + 10;
}