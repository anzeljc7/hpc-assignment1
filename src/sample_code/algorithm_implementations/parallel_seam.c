#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "utils.h"
#include "stb_image.h" 

// Algoritem
// 1. KORAK: Paralelni izračun energije (Sobel)
void calculate_energy(const unsigned char *img, float *energy, int w, int h, int orig_w, int cpp) {
    #pragma omp parallel for
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            float total_energy = 0.0f;

            for (int ch = 0; ch < cpp; ch++) {
                // Branje sosednjih pikslov (get_pixel zdaj pride iz utils.h)
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

// 2. KORAK (OPTIMIZIRAN): Izračun kumulativne energije s pomočjo trikotnikov
void calculate_cumulative_energy_triangles(const float *energy, float *cumulative, int w, int h, int orig_w, int B) {
    // Inicializacija spodnje vrstice
    #pragma omp parallel for
    for (int c = 0; c < w; c++) {
        cumulative[(h - 1) * orig_w + c] = energy[(h - 1) * orig_w + c];
    }

    // Grem od spodaj navzgor, po pasovih (strips) višine B
    for (int r_bottom = h - 2; r_bottom >= 0; r_bottom -= B) {
        
        // Zgornja meja trenutnega pasu
        int r_top = r_bottom - B + 1;
        if (r_top < 0) r_top = 0; // Zaščita na vrhu slike, če se pas ne izide točno
        
        int current_B = r_bottom - r_top + 1;

        // Izračunamo, koliko trikotnikov potrebujemo, da pokrijemo širino slike
        int num_triangles = (w + 2 * current_B - 1) / (2 * current_B) + 1;

        // --- KORAK A: Izračun NAVZGOR obrnjenih trikotnikov ---
        #pragma omp parallel for schedule(dynamic)
        for (int t = 0; t < num_triangles; t++) {
            // Vsak trikotnik ima svoj "vrh" na neki x koordinati
            int c_peak = t * 2 * current_B;

            for (int r = r_bottom; r >= r_top; r--) {
                // Širina trikotnika raste, ko se premikamo navzdol stran od vrha
                int spread = r - r_top; 
                int c_start = c_peak - spread;
                int c_end = c_peak + spread;

                // Omejimo na meje slike
                if (c_start < 0) c_start = 0;
                if (c_end >= w) c_end = w - 1;

                // Računamo vrstico znotraj trikotnika
                for (int c = c_start; c <= c_end; c++) {
                    float m_left  = (c > 0)     ? cumulative[(r + 1) * orig_w + c - 1] : 1e9f;
                    float m_mid   =               cumulative[(r + 1) * orig_w + c];
                    float m_right = (c < w - 1) ? cumulative[(r + 1) * orig_w + c + 1] : 1e9f;

                    float min_m = m_mid;
                    if (m_left < min_m) min_m = m_left;
                    if (m_right < min_m) min_m = m_right;

                    cumulative[r * orig_w + c] = energy[r * orig_w + c] + min_m;
                }
            }
        } // Implicitna OpenMP sinhronizacija (barrier) tukaj!

        // --- KORAK B: Izračun NAVZDOL obrnjenih trikotnikov (zapolnjevanje vrzeli) ---
        #pragma omp parallel for schedule(dynamic)
        for (int t = 0; t < num_triangles; t++) {
            for (int r = r_bottom; r >= r_top; r--) {
                // Vrzeli se računajo točno MED navzgor obrnjenimi trikotniki
                int c_start = t * 2 * current_B + (r - r_top) + 1;
                int c_end = (t + 1) * 2 * current_B - (r - r_top) - 1;

                // Omejimo na meje slike
                if (c_start < 0) c_start = 0;
                if (c_end >= w) c_end = w - 1;

                // Če ni vrzeli za zapolniti, zanka ne naredi nič
                for (int c = c_start; c <= c_end; c++) {
                    float m_left  = (c > 0)     ? cumulative[(r + 1) * orig_w + c - 1] : 1e9f;
                    float m_mid   =               cumulative[(r + 1) * orig_w + c];
                    float m_right = (c < w - 1) ? cumulative[(r + 1) * orig_w + c + 1] : 1e9f;

                    float min_m = m_mid;
                    if (m_left < min_m) min_m = m_left;
                    if (m_right < min_m) min_m = m_right;

                    cumulative[r * orig_w + c] = energy[r * orig_w + c] + min_m;
                }
            }
        } // Implicitna OpenMP sinhronizacija (barrier) tukaj!
    }
}

// Iskanje K šivov hkrati
void find_multiple_seams(float *cumulative, int *seams, int w, int h, int orig_w, int k) {
    for (int i = 0; i < k; i++) {
        float min_val = 1e9f;
        int min_c = 0;

        // Najdemo minimum v zgornji vrstici
        for (int c = 0; c < w; c++) {
            if (cumulative[0 * orig_w + c] < min_val) {
                min_val = cumulative[0 * orig_w + c];
                min_c = c;
            }
        }
        
        seams[i * h + 0] = min_c;
        cumulative[0 * orig_w + min_c] = 1e9f; // Označimo kot uporabljeno

        // Sledimo poti navzdol
        for (int r = 0; r < h - 1; r++) {
            int c = seams[i * h + r];

            float m_left  = (c > 0)     ? cumulative[(r + 1) * orig_w + c - 1] : 1e9f;
            float m_mid   =               cumulative[(r + 1) * orig_w + c];
            float m_right = (c < w - 1) ? cumulative[(r + 1) * orig_w + c + 1] : 1e9f;

            int next_c = c;
            float min_m = m_mid;

            if (m_left < min_m)  { min_m = m_left;  next_c = c - 1; }
            if (m_right < min_m) { min_m = m_right; next_c = c + 1; }

            seams[i * h + r + 1] = next_c;
            
            // "Zastrupimo" ta piksel in njegove neposredne sosede, da preprečimo križanje šivov
            cumulative[(r + 1) * orig_w + next_c] = 1e9f;
            if (next_c > 0) cumulative[(r + 1) * orig_w + next_c - 1] = 1e9f;
            if (next_c < w - 1) cumulative[(r + 1) * orig_w + next_c + 1] = 1e9f;
        }
    }
}

// Odstranjevanje K šivov hkrati z uporabo maske
void remove_multiple_seams(unsigned char *img, int *seams, unsigned char *mask, int w, int h, int orig_w, int cpp, int k) {
    // 1. Resetiramo masko na 0
    #pragma omp parallel for
    for (int i = 0; i < h * orig_w; i++) {
        mask[i] = 0;
    }

    // 2. Označimo piksle za odstranitev (1 = odstrani)
    #pragma omp parallel for
    for (int r = 0; r < h; r++) {
        for (int i = 0; i < k; i++) {
            mask[r * orig_w + seams[i * h + r]] = 1;
        }
    }

    // 3. Vzporedno stisnemo posamezne vrstice (preskočimo odstranjene)
    #pragma omp parallel for
    for (int r = 0; r < h; r++) {
        int dst_c = 0;
        for (int c = 0; c < w; c++) {
            if (mask[r * orig_w + c] == 0) {
                if (dst_c != c) {
                    for (int ch = 0; ch < cpp; ch++) {
                        img[(r * orig_w + dst_c) * cpp + ch] = img[(r * orig_w + c) * cpp + ch];
                    }
                }
                dst_c++;
            }
        }
    }
}

// 5. Funkcija, ki dejansko poganja algoritem in meri čas (Benchmark)
// 5. Spremenjena deklaracija funkcije
void run_benchmark(unsigned char *image_in, unsigned char *working_img, float *energy, 
                   float *cumulative, int *seams, unsigned char *mask, 
                   int width, int height, int cpp, int num_seams, size_t datasize, int K) {
    
    // Kopiramo originalno sliko v delovni pomnilnik
    memcpy(working_img, image_in, datasize);
    int current_width = width;
    
    // ZBRISANO: int K = 8; in alokacije malloc, ker to zdaj pride iz argumentov!

    // Začetek merjenja časa
    double start = omp_get_wtime();
    
    for (int s = 0; s < num_seams; s += K) {
        // Če je do konca ostalo manj kot K šivov, jih odstranimo le toliko
        int current_k = (s + K <= num_seams) ? K : (num_seams - s);

        calculate_energy(working_img, energy, current_width, height, width, cpp);
        calculate_cumulative_energy_triangles(energy, cumulative, current_width, height, width, 32);
        
        find_multiple_seams(cumulative, seams, current_width, height, width, current_k);
        remove_multiple_seams(working_img, seams, mask, current_width, height, width, cpp, current_k);
        
        current_width -= current_k; // Slika je za current_k pikslov ožja
    }

    // Konec merjenja časa
    double stop = omp_get_wtime();
    double elapsed = stop - start;
    
    // Izpis čistega časa obdelave
    printf("Time: %f s\n", elapsed);
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

    // 1. Nalaganje slike (klic iz utils.c)
    int width, height, cpp;
    unsigned char *image_in = load_image_and_check(image_in_name, &width, &height, &cpp, num_seams);
    
    // 2. Izpis informacij o nitih (klic iz utils.c)
    print_omp_info();

    // 3. Alokacija spomina
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *working_img = (unsigned char *)malloc(datasize);
    float *energy = (float *)malloc(width * height * sizeof(float));
    float *cumulative = (float *)malloc(width * height * sizeof(float));

    int K = 8; // Število šivov, ki jih odstranimo hkrati
    int *seams = (int *)malloc(K * height * sizeof(int));
    unsigned char *mask = (unsigned char *)malloc(width * height * sizeof(unsigned char));

    if (!working_img || !energy || !cumulative || !seams) {
        printf("Error: Failed to allocate memory!\n");
        exit(EXIT_FAILURE);
    }

    // 4. Izvajanje algoritma
    run_benchmark(image_in, working_img, energy, cumulative, seams, mask, width, height, cpp, num_seams, datasize, K);
    // 5. Zlaganje slike skupaj in shranjevanje (klici iz utils.c)
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
    free(seams);
    free(mask);

    return 0;
}