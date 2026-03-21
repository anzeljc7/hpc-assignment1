// utils.c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sched.h>
#include <numa.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "utils.h"

unsigned char* load_image_and_check(const char *image_in_name, int *width, int *height, int *cpp, int num_seams) {
    unsigned char *image_in = stbi_load(image_in_name, width, height, cpp, COLOR_CHANNELS);
    if (image_in == NULL) {
        printf("Error loading image %s!\n", image_in_name);
        exit(EXIT_FAILURE);
    }
    if (num_seams >= *width) {
        printf("Error: num_seams must be less than image width!\n");
        stbi_image_free(image_in);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d with %d channels. Removing %d seams.\n", 
           image_in_name, *width, *height, *cpp, num_seams);
    return image_in;
}

unsigned char* repack_image(unsigned char *working_img, int width, int final_width, int height, int cpp) {
    unsigned char *image_out = (unsigned char *)malloc(final_width * height * cpp * sizeof(unsigned char));
    if (!image_out) return NULL;
    for (int r = 0; r < height; r++) {
        memcpy(&image_out[r * final_width * cpp], &working_img[r * width * cpp], final_width * cpp);
    }
    return image_out;
}

void save_image(const char *image_out_name, unsigned char *image_out, int final_width, int height, int cpp) {
    const char *file_type = strrchr(image_out_name, '.');
    if (file_type == NULL) {
        printf("Error: No file extension found!\n");
        exit(EXIT_FAILURE);
    }
    file_type++;
    if (!strcmp(file_type, "png"))
        stbi_write_png(image_out_name, final_width, height, cpp, image_out, final_width * cpp);
    else if (!strcmp(file_type, "jpg"))
        stbi_write_jpg(image_out_name, final_width, height, cpp, image_out, 100);
    else if (!strcmp(file_type, "bmp"))
        stbi_write_bmp(image_out_name, final_width, height, cpp, image_out);
    else
        printf("Error: Unknown image format %s!\n", file_type);
}

void print_omp_info() {
    #pragma omp parallel
    {
        #pragma omp single
        printf("Using %d threads.\n", omp_get_num_threads());

        int tid = omp_get_thread_num();
        int cpu = sched_getcpu();
        int node = numa_node_of_cpu(cpu);

        #pragma omp critical
        {
            if (tid == 0 || tid == omp_get_num_threads() - 1) {
                printf("Thread %d -> CPU %d NUMA %d\n", tid, cpu, node);
            }
        }
    }
}