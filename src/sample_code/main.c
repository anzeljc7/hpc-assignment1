#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <sched.h>
#include <numa.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 0
#define MAX_FILENAME 255

// Pomožna funkcija za branje vrednosti piksla (z omejitvijo na robovih slike)
inline int get_pixel(const unsigned char *img, int r, int c, int w, int h, int orig_w, int ch, int cpp) {
    if (r < 0) r = 0;
    if (r >= h) r = h - 1;
    if (c < 0) c = 0;
    if (c >= w) c = w - 1;
    return img[(r * orig_w + c) * cpp + ch];
}

// 1. KORAK: Paralelni izračun energije (Sobel)
void calculate_energy(const unsigned char *img, float *energy, int w, int h, int orig_w, int cpp) {
    #pragma omp parallel for
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            float total_energy = 0.0f;

            for (int ch = 0; ch < cpp; ch++) {
                // Branje sosednjih pikslov
                int p_tl = get_pixel(img, r - 1, c - 1, w, h, orig_w, ch, cpp);
                int p_tc = get_pixel(img, r - 1, c,     w, h, orig_w, ch, cpp);
                int p_tr = get_pixel(img, r - 1, c + 1, w, h, orig_w, ch, cpp);
                
                int p_ml = get_pixel(img, r,     c - 1, w, h, orig_w, ch, cpp);
                int p_mr = get_pixel(img, r,     c + 1, w, h, orig_w, ch, cpp);
                
                int p_bl = get_pixel(img, r + 1, c - 1, w, h, orig_w, ch, cpp);
                int p_bc = get_pixel(img, r + 1, c,     w, h, orig_w, ch, cpp);
                int p_br = get_pixel(img, r + 1, c + 1, w, h, orig_w, ch, cpp);

                // Gx = zaznavanje horizontalnih sprememb
                float gx = -p_tl - 2 * p_ml - p_bl + p_tr + 2 * p_mr + p_br;
                // Gy = zaznavanje vertikalnih sprememb
                float gy = p_tl + 2 * p_tc + p_tr - p_bl - 2 * p_bc - p_br;

                total_energy += sqrtf(gx * gx + gy * gy);
            }
            // Povprečje po vseh kanalih
            energy[r * orig_w + c] = total_energy / (float)cpp;
        }
    }
}

// 2. KORAK: Paralelni izračun kumulativne energije (od spodaj navzgor)
void calculate_cumulative_energy(const float *energy, float *cumulative, int w, int h, int orig_w) {
    // Inicializacija spodnje vrstice
    #pragma omp parallel for
    for (int c = 0; c < w; c++) {
        cumulative[(h - 1) * orig_w + c] = energy[(h - 1) * orig_w + c];
    }

    // Izračun vrstico po vrstico od spodaj navzgor z implicitno sinhronizacijo
    for (int r = h - 2; r >= 0; r--) {
        #pragma omp parallel for
        for (int c = 0; c < w; c++) {
            float m_left  = (c > 0)     ? cumulative[(r + 1) * orig_w + c - 1] : 1e9f;
            float m_mid   =               cumulative[(r + 1) * orig_w + c];
            float m_right = (c < w - 1) ? cumulative[(r + 1) * orig_w + c + 1] : 1e9f;

            float min_m = m_mid;
            if (m_left < min_m) min_m = m_left;
            if (m_right < min_m) min_m = m_right;

            cumulative[r * orig_w + c] = energy[r * orig_w + c] + min_m;
        }
    }
}

// 3. KORAK: Sekvenčno iskanje poti šiva od zgoraj navzdol
void find_seam(const float *cumulative, int *seam, int w, int h, int orig_w) {
    float min_val = 1e9f;
    int min_c = 0;

    // Najdemo minimum v zgornji vrstici
    for (int c = 0; c < w; c++) {
        if (cumulative[0 * orig_w + c] < min_val) {
            min_val = cumulative[0 * orig_w + c];
            min_c = c;
        }
    }
    seam[0] = min_c;

    // Sledimo poti navzdol
    for (int r = 0; r < h - 1; r++) {
        int c = seam[r];

        float m_left  = (c > 0)     ? cumulative[(r + 1) * orig_w + c - 1] : 1e9f;
        float m_mid   =               cumulative[(r + 1) * orig_w + c];
        float m_right = (c < w - 1) ? cumulative[(r + 1) * orig_w + c + 1] : 1e9f;

        int next_c = c;
        float min_m = m_mid;

        if (m_left < min_m) {
            min_m = m_left;
            next_c = c - 1;
        }
        if (m_right < min_m) {
            min_m = m_right;
            next_c = c + 1;
        }

        seam[r + 1] = next_c;
    }
}

// 4. KORAK: Paralelni pomik pikslov na levi po odstranitvi šiva
void remove_seam(unsigned char *img, const int *seam, int w, int h, int orig_w, int cpp) {
    #pragma omp parallel for
    for (int r = 0; r < h; r++) {
        int c_remove = seam[r];
        if (c_remove < w - 1) {
            // Premaknemo piksle na desni strani šiva za eno mesto v levo
            unsigned char *dst = &img[(r * orig_w + c_remove) * cpp];
            unsigned char *src = &img[(r * orig_w + c_remove + 1) * cpp];
            size_t bytes = (w - 1 - c_remove) * cpp;
            memmove(dst, src, bytes);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("USAGE: %s input_image output_image num_seams\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char image_in_name[MAX_FILENAME];
    char image_out_name[MAX_FILENAME];
    snprintf(image_in_name, MAX_FILENAME, "%s", argv[1]);
    snprintf(image_out_name, MAX_FILENAME, "%s", argv[2]);
    int num_seams = atoi(argv[3]);

    // Load image from file
    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (image_in == NULL) {
        printf("Error reading loading image %s!\n", image_in_name);
        exit(EXIT_FAILURE);
    }
    
    if (num_seams >= width) {
        printf("Error: num_seams must be less than image width!\n");
        stbi_image_free(image_in);
        exit(EXIT_FAILURE);
    }

    printf("Loaded image %s of size %dx%d with %d channels. Removing %d seams.\n", image_in_name, width, height, cpp, num_seams);
    
    // Izpis niti (kot v originalnem sample.c)
    #pragma omp parallel
    {
        #pragma omp single
        printf("Using %d threads.\n", omp_get_num_threads());

        int tid = omp_get_thread_num();
        int cpu = sched_getcpu();
        int node = numa_node_of_cpu(cpu);

        #pragma omp critical
        {
            // Izpisano samo za prve in zadnje niti, da terminal ne bo prenatrpan pri 32 nitih
            if (tid == 0 || tid == omp_get_num_threads() - 1) {
                printf("Thread %d -> CPU %d NUMA %d\n", tid, cpu, node);
            }
        }
    }

    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *working_img = (unsigned char *)malloc(datasize);
    
    float *energy = (float *)malloc(width * height * sizeof(float));
    float *cumulative = (float *)malloc(width * height * sizeof(float));
    int *seam = (int *)malloc(height * sizeof(int));

    if (!working_img || !energy || !cumulative || !seam) {
        printf("Error: Failed to allocate memory!\n");
        exit(EXIT_FAILURE);
    }

    int NUM_RUNS = 5;
    double total_time = 0.0;
    int final_width = width - num_seams;

    for (int run = 0; run < NUM_RUNS; run++) {
        // Obnovimo originalno sliko na začetku vsakega testa
        memcpy(working_img, image_in, datasize);
        int current_width = width;

        double start = omp_get_wtime();
        
        for (int s = 0; s < num_seams; s++) {
            calculate_energy(working_img, energy, current_width, height, width, cpp);
            calculate_cumulative_energy(energy, cumulative, current_width, height, width);
            find_seam(cumulative, seam, current_width, height, width);
            remove_seam(working_img, seam, current_width, height, width, cpp);
            current_width--;
        }

        double stop = omp_get_wtime();
        double elapsed = stop - start;
        printf("Run %d Time: %f s\n", run + 1, elapsed);
        total_time += elapsed;
    }

    printf("\nAverage Time over %d runs: %f s\n", NUM_RUNS, total_time / NUM_RUNS);

    // Po odstranjevanju šivov slika v spominu vsebuje "luknje" na koncu vsake vrstice.
    // Piksle zložimo skupaj, da bo izhodna slika pravilnega formata.
    unsigned char *image_out = (unsigned char *)malloc(final_width * height * cpp * sizeof(unsigned char));
    for (int r = 0; r < height; r++) {
        memcpy(&image_out[r * final_width * cpp], &working_img[r * width * cpp], final_width * cpp);
    }

    // Write the output image to file
    char image_out_name_temp[MAX_FILENAME];
    strncpy(image_out_name_temp, image_out_name, MAX_FILENAME);

    const char *file_type = strrchr(image_out_name, '.');
    if (file_type == NULL) {
        printf("Error: No file extension found!\n");
        exit(EXIT_FAILURE);
    }
    file_type++; // skip the dot

    if (!strcmp(file_type, "png"))
        stbi_write_png(image_out_name, final_width, height, cpp, image_out, final_width * cpp);
    else if (!strcmp(file_type, "jpg"))
        stbi_write_jpg(image_out_name, final_width, height, cpp, image_out, 100);
    else if (!strcmp(file_type, "bmp"))
        stbi_write_bmp(image_out_name, final_width, height, cpp, image_out);
    else
        printf("Error: Unknown image format %s! Only png, jpg, or bmp supported.\n", file_type);

    // Release the memory
    stbi_image_free(image_in);
    free(working_img);
    free(image_out);
    free(energy);
    free(cumulative);
    free(seam);

    return 0;
}