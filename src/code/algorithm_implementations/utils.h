#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

#define COLOR_CHANNELS 0
#define MAX_FILENAME 255

// Pomožna funkcija za branje piksla
static inline int get_pixel(const unsigned char *img, int r, int c, int w, int h, int orig_w, int ch, int cpp) {
    if (c < 0) c = 0;
    if (c >= w) c = w - 1;
    if (r < 0) r = 0;
    if (r >= h) r = h - 1;
    return img[(r * orig_w + c) * cpp + ch];
}

// Splošne funkcije
unsigned char* load_image_and_check(const char *image_in_name, int *width, int *height, int *cpp, int num_seams);
unsigned char* repack_image(unsigned char *working_img, int width, int final_width, int height, int cpp);
void save_image(const char *image_out_name, unsigned char *image_out, int final_width, int height, int cpp);
void print_omp_info();

#endif