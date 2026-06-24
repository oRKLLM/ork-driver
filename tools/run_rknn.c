/* tools/run_rknn.c — Standard C-API runner for compiled .rknn models
 * 
 * Compiles and runs on the board:
 *   gcc -O2 -o run_rknn tools/run_rknn.c -Itools/ -I/home/michael/ork-driver/ -L/home/michael/ork-driver/ -lrknnrt
 *   sudo ./run_rknn sigmoid.rknn
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_api.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.rknn>\n", argv[0]);
        return 1;
    }
    rknn_context ctx;
    
    // Load the model
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) {
        perror("Failed to open model file");
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void *model_data = malloc(size);
    if (!model_data) {
        perror("malloc failed");
        fclose(fp);
        return 1;
    }
    fread(model_data, 1, size, fp);
    fclose(fp);

    printf("[run_rknn] Initializing RKNN context...\n");
    int ret = rknn_init(&ctx, model_data, size, 0, NULL);
    if (ret < 0) {
        fprintf(stderr, "rknn_init failed with error code %d\n", ret);
        free(model_data);
        return 1;
    }

    // Set up a deterministic quantized INT8 input ramp spanning [-128, 127]
    int8_t input_data[512];
    for (int i = 0; i < 512; i++) {
        float x = -4.0f + (8.0f * i / 511);
        int32_t q = (int32_t)(x * (127.0f / 4.0f));
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        input_data[i] = (int8_t)q;
    }

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_INT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = 512;
    inputs[0].buf = input_data;
    inputs[0].pass_through = 1; // Tell SDK to bypass preprocessing and pass raw INT8 bytes directly!

    printf("[run_rknn] Setting inputs...\n");
    ret = rknn_inputs_set(ctx, 1, inputs);
    if (ret < 0) {
        fprintf(stderr, "rknn_inputs_set failed: %d\n", ret);
        rknn_destroy(ctx);
        free(model_data);
        return 1;
    }
    
    printf("[run_rknn] Running inference on the NPU...\n");
    ret = rknn_run(ctx, NULL);
    if (ret < 0) {
        fprintf(stderr, "rknn_run failed with error code %d\n", ret);
        rknn_destroy(ctx);
        free(model_data);
        return 1;
    }

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 0; // We want raw quantized INT8/uint8_t outputs back!

    printf("[run_rknn] Getting outputs...\n");
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret < 0) {
        fprintf(stderr, "rknn_outputs_get failed: %d\n", ret);
    } else {
        uint8_t *out_buf = (uint8_t *)outputs[0].buf;
        printf("[run_rknn] Raw Output Bytes (first 16 elements):\n");
        for (int i = 0; i < 16; i++) {
            printf("  i=%02d | raw: %3d\n", i, out_buf[i]);
        }
        rknn_outputs_release(ctx, 1, outputs);
    }
    
    printf("[run_rknn] Tearing down...\n");
    rknn_destroy(ctx);
    free(model_data);
    return 0;
}
