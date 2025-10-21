#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/adc.h"
#include "dht11-pico.h"

// TinyML
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "qdnn_fan_model.h"
#include "qdnn_pump_model.h"

// --- Tensor arena ---
constexpr int kArenaSize = 16 * 1024;
static uint8_t fan_arena[kArenaSize];
static uint8_t pump_arena[kArenaSize];

// --- Pin definitions ---
#define DHT_PIN        3
#define SOIL_ADC_PIN   26
const uint FAN_LEDS[4]  = {10, 11, 12, 13};
const uint PUMP_LEDS[4] = {14, 15, 16, 17};
#define LED_DHT_ERROR 18

// --- Soil moisture calibration ---
#define SOIL_DRY_RAW 4000
#define SOIL_WET_RAW 1000

// --- Fungsi ADC averaging ---
uint16_t read_soil_adc() {
    const int samples = 64;
    uint32_t sum = 0;
    for (int i=0; i<samples; i++) {
        sum += adc_read();
        sleep_us(50);
    }
    return (uint16_t)(sum / samples);
}

// --- Konversi ADC ke persentase ---
float adc_to_percent(uint16_t raw) {
    float percent = (float)(SOIL_DRY_RAW - raw) * 100.0f / (SOIL_DRY_RAW - SOIL_WET_RAW);
    if (percent < 0) percent = 0;
    if (percent > 100) percent = 100;
    return percent;
}

// --- Nyalakan LED sesuai level ---
void set_led_level(const uint leds[4], int level) {
    if (level < 0) level = 0;
    if (level > 4) level = 4;
    for (int i=0;i<4;i++) gpio_put(leds[i], i<level);
}

// --- Debug: tampil info tensor ---
void print_tensor_info(TfLiteTensor* t, const char* name) {
    printf("Tensor %s: type=%d dims=", name, t->type);
    for (int i=0;i<t->dims->size;i++) printf("%d ", t->dims->data[i]);
    printf("\n");
    if (t->params.scale != 0.0f || t->params.zero_point != 0)
        printf("  quant params: scale=%.6f zero_point=%d\n", t->params.scale, t->params.zero_point);
}

// --- Jalankan model dengan auto quantization ---
int run_model_safe(tflite::MicroInterpreter &interpreter, TfLiteTensor* input_tensor,
                   TfLiteTensor* output_tensor, float* input_vals, int n_input,
                   float* out_scores_buffer, int max_out_scores) {

    print_tensor_info(input_tensor, "input");
    print_tensor_info(output_tensor, "output");

    // isi input
    if (input_tensor->type == kTfLiteFloat32) {
        for (int i=0;i<n_input;i++) input_tensor->data.f[i] = input_vals[i];
    } else if (input_tensor->type == kTfLiteInt8) {
        float scale = input_tensor->params.scale;
        int zp = input_tensor->params.zero_point;
        for (int i=0;i<n_input;i++) {
            int32_t q = (int32_t)round(input_vals[i]/scale)+zp;
            if (q<-128) q=-128; if(q>127) q=127;
            input_tensor->data.int8[i] = (int8_t)q;
        }
    } else if (input_tensor->type == kTfLiteUInt8) {
        float scale = input_tensor->params.scale;
        int zp = input_tensor->params.zero_point;
        for (int i=0;i<n_input;i++) {
            int32_t q = (int32_t)round(input_vals[i]/scale)+zp;
            if (q<0) q=0; if(q>255) q=255;
            input_tensor->data.uint8[i] = (uint8_t)q;
        }
    } else {
        printf("Unsupported input tensor type %d\n", input_tensor->type);
        return -1;
    }

    if(interpreter.Invoke()!=kTfLiteOk) { printf("Invoke failed\n"); return -1; }

    // baca output dan dequantize
    int out_classes = output_tensor->dims->data[1];
    if(out_classes>max_out_scores) out_classes = max_out_scores;

    if(output_tensor->type==kTfLiteFloat32) {
        for(int i=0;i<out_classes;i++) out_scores_buffer[i] = output_tensor->data.f[i];
    } else if(output_tensor->type==kTfLiteInt8) {
        float scale = output_tensor->params.scale;
        int zp = output_tensor->params.zero_point;
        for(int i=0;i<out_classes;i++)
            out_scores_buffer[i] = (output_tensor->data.int8[i]-zp)*scale;
    } else if(output_tensor->type==kTfLiteUInt8) {
        float scale = output_tensor->params.scale;
        int zp = output_tensor->params.zero_point;
        for(int i=0;i<out_classes;i++)
            out_scores_buffer[i] = (output_tensor->data.uint8[i]-zp)*scale;
    } else {
        printf("Unsupported output tensor type %d\n", output_tensor->type);
        return -1;
    }

    // prediksi kelas
    int pred = 0;
    float maxs = out_scores_buffer[0];
    for(int i=1;i<out_classes;i++) {
        if(out_scores_buffer[i]>maxs) { maxs=out_scores_buffer[i]; pred=i; }
    }

    // print scores
    printf("Scores: ");
    for(int i=0;i<out_classes;i++) printf("%.3f ", out_scores_buffer[i]);
    printf("\n");

    return pred;
}

int main() {
    stdio_init_all();
    sleep_ms(2000);
    printf("=== Pico DHT11 + Soil + TinyML Fan/Pump ===\n");

    // --- Init GPIO ---
    for(int i=0;i<4;i++){
        gpio_init(FAN_LEDS[i]); gpio_set_dir(FAN_LEDS[i], GPIO_OUT); gpio_put(FAN_LEDS[i],0);
        gpio_init(PUMP_LEDS[i]); gpio_set_dir(PUMP_LEDS[i], GPIO_OUT); gpio_put(PUMP_LEDS[i],0);
    }
    gpio_init(LED_DHT_ERROR); gpio_set_dir(LED_DHT_ERROR, GPIO_OUT); gpio_put(LED_DHT_ERROR,0);

    // --- Init ADC ---
    adc_init();
    adc_gpio_init(SOIL_ADC_PIN);
    adc_select_input(0);

    // --- DHT ---
    float temp=0, humid=0;

    // --- TinyML Fan ---
    const tflite::Model* fan_model = tflite::GetModel(qdnn_fan_model);
    tflite::MicroMutableOpResolver<10> fan_resolver;
    fan_resolver.AddFullyConnected(); fan_resolver.AddReshape(); fan_resolver.AddSoftmax();
    tflite::MicroInterpreter fan_interpreter(fan_model, fan_resolver, fan_arena, kArenaSize);
    fan_interpreter.AllocateTensors();
    TfLiteTensor* fan_input = fan_interpreter.input(0);
    TfLiteTensor* fan_output = fan_interpreter.output(0);

    // --- TinyML Pump ---
    const tflite::Model* pump_model = tflite::GetModel(qdnn_pump_model);
    tflite::MicroMutableOpResolver<10> pump_resolver;
    pump_resolver.AddFullyConnected(); pump_resolver.AddReshape(); pump_resolver.AddSoftmax();
    tflite::MicroInterpreter pump_interpreter(pump_model, pump_resolver, pump_arena, kArenaSize);
    pump_interpreter.AllocateTensors();
    TfLiteTensor* pump_input = pump_interpreter.input(0);
    TfLiteTensor* pump_output = pump_interpreter.output(0);

    float scores_buf[16];

    while(true){
        // --- Baca DHT11 ---
        int status = read_from_dht(DHT_PIN,&temp,&humid,false);
        if(status!=0 || humid<0 || humid>100){gpio_put(LED_DHT_ERROR,1); sleep_ms(2000); continue;}
        else gpio_put(LED_DHT_ERROR,0);

        // --- Soil ---
        uint16_t raw = read_soil_adc();
        float soil_pct = adc_to_percent(raw);

        // --- Siapkan input model ---
        float ml_input[3]={temp, humid, soil_pct};

        // --- Fan ---
        int fan_level = run_model_safe(fan_interpreter, fan_input, fan_output, ml_input, 3, scores_buf,16);
        set_led_level(FAN_LEDS,fan_level);

        // --- Pump ---
        int pump_level = run_model_safe(pump_interpreter, pump_input, pump_output, ml_input, 3, scores_buf,16);
        set_led_level(PUMP_LEDS,pump_level);

        // --- Print hasil ---
        printf("-----------------------------------\n");
        printf("Temp: %.1f°C | Humid: %.1f%% | Soil: %.1f%% (raw=%u)\n", temp, humid, soil_pct, raw);
        printf("Fan Level: %d | Pump Level: %d\n", fan_level, pump_level);

        sleep_ms(3000);
    }
}
