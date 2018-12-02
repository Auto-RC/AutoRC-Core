#include <SPI.h>
#include <nRF24L01.h>
#include <RF24.h>

RF24 radio(7, 8); // CE, CSN
const byte address[6] = "00001";
int len = 0;
float turn = 0;
float throttle = 0;
float nums[] = {0, 0};

void setup() {
  Serial.begin(250000);
  radio.begin();
  radio.openWritingPipe(address);
  radio.setPALevel(RF24_PA_MAX);
  radio.stopListening();
  pinMode(2, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0){
    len += 1;
    if (len <= 4){
      int d = Serial.read() - '0';
      turn += d * pow(10, 4-len);
    } else if (len <= 8){
      int d = Serial.read() - '0';
      throttle += d * pow(10, 4-(len-4));
    }
  }
  if (len == 8){
    Serial.println(turn);
    Serial.println(throttle);
    turn /= 10000;
    throttle /= 10000;
    nums[0] = turn;
    nums[1] = throttle;
    Serial.println(turn);
    Serial.println(throttle);
    radio.write(&nums, sizeof(nums));
    digitalWrite(2, HIGH);
    len = 0;
    turn = 0;
    throttle = 0;
  }

}

