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
    if (c < 0) c = 0;
    if (c >= w) c = w - 1;
    if (r < 0) r = 0;
    if (r >= h) r = h - 1;
    return img[(r * orig_w + c) * cpp + ch];
}

// Algoritem
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

// Ostale pomožne funkicje
// 1. Funkcija za nalaganje slike
unsigned char* load_image_and_check(const char *image_in_name, int *width, int *height, int *cpp, int num_seams) {
    // Klic knjižnice stb (upoštevaj, da width, height in cpp že podajamo kot kazalce)
    unsigned char *image_in = stbi_load(image_in_name, width, height, cpp, COLOR_CHANNELS);

    if (image_in == NULL) {
        printf("Error loading image %s!\n", image_in_name);
        exit(EXIT_FAILURE);
    }
    
    // Uporabimo *width, ker je width zdaj kazalec
    if (num_seams >= *width) {
        printf("Error: num_seams must be less than image width!\n");
        stbi_image_free(image_in);
        exit(EXIT_FAILURE);
    }

    printf("Loaded image %s of size %dx%d with %d channels. Removing %d seams.\n", 
           image_in_name, *width, *height, *cpp, num_seams);

    return image_in;
}

// 2. Funkcija za izpis OpenMP diagnostike
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

// 3. Funkcija, ki dejansko poganja algoritem in meri čas (Benchmark)
void run_benchmark(unsigned char *image_in, unsigned char *working_img, float *energy, 
                   float *cumulative, int *seam, int width, int height, int cpp, int num_seams, size_t datasize) {
    
    // Kopiramo originalno sliko v delovni pomnilnik
    memcpy(working_img, image_in, datasize);
    int current_width = width;
    
    // Začetek merjenja časa
    double start = omp_get_wtime();
    
    for (int s = 0; s < num_seams; s++) {
        calculate_energy(working_img, energy, current_width, height, width, cpp);
        calculate_cumulative_energy(energy, cumulative, current_width, height, width);
        find_seam(cumulative, seam, current_width, height, width);
        remove_seam(working_img, seam, current_width, height, width, cpp);
        current_width--; // Slika je zdaj za 1 piksel ožja
    }

    // Konec merjenja časa
    double stop = omp_get_wtime();
    double elapsed = stop - start;
    
    // Izpis čistega časa obdelave
    printf("Time: %f s\n", elapsed);
}

// 4. Funkcija za odstranjevanje "lukenj" v spominu po izrezu šivov
unsigned char* repack_image(unsigned char *working_img, int width, int final_width, int height, int cpp) {
    unsigned char *image_out = (unsigned char *)malloc(final_width * height * cpp * sizeof(unsigned char));
    if (!image_out) return NULL;

    for (int r = 0; r < height; r++) {
        memcpy(&image_out[r * final_width * cpp], &working_img[r * width * cpp], final_width * cpp);
    }
    return image_out;
}

// 5. Funkcija za shranjevanje slike v ustreznem formatu
void save_image(const char *image_out_name, unsigned char *image_out, int final_width, int height, int cpp) {
    const char *file_type = strrchr(image_out_name, '.');
    if (file_type == NULL) {
        printf("Error: No file extension found!\n");
        exit(EXIT_FAILURE);
    }
    file_type++; // preskoči piko

    if (!strcmp(file_type, "png"))
        stbi_write_png(image_out_name, final_width, height, cpp, image_out, final_width * cpp);
    else if (!strcmp(file_type, "jpg"))
        stbi_write_jpg(image_out_name, final_width, height, cpp, image_out, 100);
    else if (!strcmp(file_type, "bmp"))
        stbi_write_bmp(image_out_name, final_width, height, cpp, image_out);
    else
        printf("Error: Unknown image format %s! Only png, jpg, or bmp supported.\n", file_type);
}

// ================= MAIN =================

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Wrong number of arguments!\n");
        exit(EXIT_FAILURE);
    }

    char image_in_name[MAX_FILENAME];
    char image_out_name[MAX_FILENAME];
    snprintf(image_in_name, MAX_FILENAME, "%s", argv[1]);
    snprintf(image_out_name, MAX_FILENAME, "%s", argv[2]);
    int num_seams = atoi(argv[3]);

    // 1. Nalaganje slike
    int width, height, cpp;
    unsigned char *image_in = load_image_and_check(image_in_name, &width, &height, &cpp, num_seams);
    
    // 2. Izpis informacij o nitih
    print_omp_info();

    // 3. Alokacija spomina
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *working_img = (unsigned char *)malloc(datasize);
    float *energy = (float *)malloc(width * height * sizeof(float));
    float *cumulative = (float *)malloc(width * height * sizeof(float));
    int *seam = (int *)malloc(height * sizeof(int));

    if (!working_img || !energy || !cumulative || !seam) {
        printf("Error: Failed to allocate memory!\n");
        exit(EXIT_FAILURE);
    }

    // 4. Izvajanje algoritma (merjenje časa je znotraj te funkcije)
    run_benchmark(image_in, working_img, energy, cumulative, seam, width, height, cpp, num_seams, datasize);

    // 5. Zlaganje slike skupaj in shranjevanje
    int final_width = width - num_seams;
    unsigned char *image_out = repack_image(working_img, width, final_width, height, cpp);
    
    if (image_out) {
        save_image(image_out_name, image_out, final_width, height, cpp);
    }

    // 6. Čiščenje spomina
    stbi_image_free(image_in);
    free(working_img);
    free(image_out);
    free(energy);
    free(cumulative);
    free(seam);

    return 0;
}