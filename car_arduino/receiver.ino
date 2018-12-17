#include <SPI.h>
#include <nRF24L01.h>
#include <RF24.h>

int led = 2;
RF24 radio(7, 8);
float onOff;
const byte address[6] = "00001";

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(led, OUTPUT);
  radio.begin();
  radio.openReadingPipe(0, address);
  radio.setPALevel(RF24_PA_MIN);
  radio.startListening();
}

void loop() {

  if (radio.available()) {
    radio.read(&onOff, sizeof(onOff));
    Serial.println(onOff);
  }


}
