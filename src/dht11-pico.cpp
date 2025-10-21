/**
 * @file dht11-pico.cpp
 *
 * @brief DHT11 Sensor Library Implementation for Raspberry Pi Pico
 *
 * This file contains the implementation of the DHT11 sensor library for Raspberry Pi Pico,
 * including the initialization, reading, and retrieval of temperature and humidity values
 * from the DHT11 sensor. It also provides a C-style function read_from_dht() for legacy calls.
 */


 #include "dht11-pico.h"
 #include <cstdio>      // <- Tambahkan ini

// =======================
// Class Dht11 Implementation
// =======================

Dht11::Dht11(uint pin){
    gpioPin = pin;
    gpio_init(pin);
    sleep_ms(1000);  // wait for sensor to stabilize
}

Dht11::~Dht11(){
    gpio_deinit(gpioPin);
}

long long Dht11::read(){
    int count = 0;
    long long raw = 0;

    // Start signal
    gpio_set_dir(gpioPin, GPIO_OUT);
    gpio_put(gpioPin, 0);
    sleep_ms(20);
    gpio_set_dir(gpioPin, GPIO_IN);

    // Wait for sensor response
    while (gpio_get(gpioPin) == 1) {
        count++;
        sleep_us(5);
        if (count == POLLING_LIMIT) return TRANSMISSION_ERROR;
    }

    count = 0;
    while (gpio_get(gpioPin) == 0) {
        count++;
        sleep_us(5);
        if (count == POLLING_LIMIT) return TRANSMISSION_ERROR;
    }

    count = 0;
    while (gpio_get(gpioPin) == 1) {
        count++;
        sleep_us(5);
        if (count == POLLING_LIMIT) return TRANSMISSION_ERROR;
    }

    // Read 40 bits
    for (int i = 0; i < 40; i++) {
        count = 0;
        while (gpio_get(gpioPin) == 0) sleep_us(5);
        while (gpio_get(gpioPin) == 1) {
            sleep_us(5);
            count++;
        }
        if (count >= THRESHOLD) raw |= 1;
        raw = raw << 1;
    }

    // Checksum validation
    if (((raw & RH_INT_MASK) >> 32) +
        ((raw & RH_DEC_MASK) >> 24) +
        ((raw & TEMP_INT_MASK) >> 16) +
        ((raw & TEMP_DEC_MASK) >> 8) -
        (raw & CHECKSUM_MASK) > 1) {
        return TRANSMISSION_ERROR;
    }

    return raw;
}

double Dht11::readT() {
    long long raw = read();
    if (raw == TRANSMISSION_ERROR) return TRANSMISSION_ERROR;

    int temp_int  = (raw & TEMP_INT_MASK) >> 16;
    uint temp_dec = (raw & TEMP_DEC_MASK) >> 8;
    if (temp_int < 0) temp_dec = -temp_dec;
    double temp = temp_int + 0.1 * temp_dec;
    return temp;
}

double Dht11::readRH() {
    long long raw = read();
    if (raw == TRANSMISSION_ERROR) return TRANSMISSION_ERROR;

    int rh_int  = (raw & RH_INT_MASK) >> 32;
    uint rh_dec = (raw & RH_DEC_MASK) >> 24;
    double rh = rh_int + 0.1 * rh_dec;
    return rh;
}

void Dht11::readRHT(double* temp, double* rh) {
    long long raw = read();
    if (raw == TRANSMISSION_ERROR) {
        *temp = *rh = TRANSMISSION_ERROR;
        return;
    }

    int temp_int  = (raw & TEMP_INT_MASK) >> 16;
    uint temp_dec = (raw & TEMP_DEC_MASK) >> 8;
    int rh_int    = (raw & RH_INT_MASK) >> 32;
    uint rh_dec   = (raw & RH_DEC_MASK) >> 24;

    *temp = temp_int + 0.1 * temp_dec;
    *rh   = rh_int + 0.1 * rh_dec;
}

// =======================
// C-style Helper Function
// =======================

int read_from_dht(uint gpio_pin, float* temperature, float* humidity, bool debug) {
    if (!temperature || !humidity) return TRANSMISSION_ERROR;

    Dht11 sensor(gpio_pin);
    *temperature = sensor.readT();
    *humidity    = sensor.readRH();

    if (debug) {
        printf("[DHT11] Read: T=%.1f, H=%.1f\n", *temperature, *humidity);
    }

    return 0;  // success
}
