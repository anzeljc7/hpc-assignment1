#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "utils.h"
#include "stb_image.h" 

void calculate_energy(const unsigned char *img, float *energy, int curr_width, int curr_height, int orig_width, int cpp) {
    #pragma omp parallel for
    for (int r = 0; r < curr_height; r++) {
        for (int c = 0; c < curr_width; c++) {
            float total_energy = 0.0f;

            for (int ch = 0; ch < cpp; ch++) {
                int p_tl = get_pixel(img, r - 1, c - 1, curr_width, curr_height, orig_width, ch, cpp);
                int p_tc = get_pixel(img, r - 1, c,     curr_width, curr_height, orig_width, ch, cpp);
                int p_tr = get_pixel(img, r - 1, c + 1, curr_width, curr_height, orig_width, ch, cpp);
                
                int p_ml = get_pixel(img, r,     c - 1, curr_width, curr_height, orig_width, ch, cpp);
                int p_mr = get_pixel(img, r,     c + 1, curr_width, curr_height, orig_width, ch, cpp);
                
                int p_bl = get_pixel(img, r + 1, c - 1, curr_width, curr_height, orig_width, ch, cpp);
                int p_bc = get_pixel(img, r + 1, c,     curr_width, curr_height, orig_width, ch, cpp);
                int p_br = get_pixel(img, r + 1, c + 1, curr_width, curr_height, orig_width, ch, cpp);

                float gx = -p_tl - 2 * p_ml - p_bl + p_tr + 2 * p_mr + p_br;
                float gy = p_tl + 2 * p_tc + p_tr - p_bl - 2 * p_bc - p_br;

                total_energy += sqrtf(gx * gx + gy * gy);
            }
            energy[r * orig_width + c] = total_energy / (float)cpp;
        }
    }
}

void calculate_cumulative_energy_triangles(const float *energy, float *cumulative, int curr_width, int curr_height, int orig_width, int B) {
    #pragma omp parallel for
    for (int c = 0; c < curr_width; c++) {
        cumulative[(curr_height - 1) * orig_width + c] = energy[(curr_height - 1) * orig_width + c];
    }

    for (int r_bottom = curr_height - 2; r_bottom >= 0; r_bottom -= B) {
        int r_top = r_bottom - B + 1;
        if (r_top < 0) r_top = 0; 
        
        int current_B = r_bottom - r_top + 1;
        int num_triangles = (curr_width + 2 * current_B - 1) / (2 * current_B) + 1;

        // KORAK A: Izračun NAVZGOR obrnjenih trikotnikov
        #pragma omp parallel for
        for (int t = 0; t < num_triangles; t++) {
            int c_peak = t * 2 * current_B;

            for (int r = r_bottom; r >= r_top; r--) {
                int spread = r_bottom - r; 
                int c_start = c_peak - spread;
                int c_end = c_peak + spread;

                if (c_start < 0) c_start = 0;
                if (c_end >= curr_width) c_end = curr_width - 1;

                for (int c = c_start; c <= c_end; c++) {
                    float m_left  = (c > 0)                  ? cumulative[(r + 1) * orig_width + c - 1] : 1e9f;
                    float m_mid   =                            cumulative[(r + 1) * orig_width + c];
                    float m_right = (c < curr_width - 1) ? cumulative[(r + 1) * orig_width + c + 1] : 1e9f;

                    float min_m = m_mid;
                    if (m_left < min_m) min_m = m_left;
                    if (m_right < min_m) min_m = m_right;

                    cumulative[r * orig_width + c] = energy[r * orig_width + c] + min_m;
                }
            }
        } 

        // KORAK B: Izračun NAVZDOL obrnjenih trikotnikov
        #pragma omp parallel for
        for (int t = 0; t < num_triangles; t++) {
            for (int r = r_bottom; r >= r_top; r--) {
                int spread = r_bottom - r;
                int c_start = t * 2 * current_B + spread + 1;
                int c_end = (t + 1) * 2 * current_B - spread - 1;

                if (c_start < 0) c_start = 0;
                if (c_end >= curr_width) c_end = curr_width - 1;

                for (int c = c_start; c <= c_end; c++) {
                    float m_left  = (c > 0)                  ? cumulative[(r + 1) * orig_width + c - 1] : 1e9f;
                    float m_mid   =                            cumulative[(r + 1) * orig_width + c];
                    float m_right = (c < curr_width - 1) ? cumulative[(r + 1) * orig_width + c + 1] : 1e9f;

                    float min_m = m_mid;
                    if (m_left < min_m) min_m = m_left;
                    if (m_right < min_m) min_m = m_right;

                    cumulative[r * orig_width + c] = energy[r * orig_width + c] + min_m;
                }
            }
        } 
    }
}

// Iskanje več šivov hkrati
// Poišče K šivov naenkrat. Da ne bi izbrali istega šiva večkrat, 
// po vsaki najdeni poti jo "zastrupimo" (ji nastavimo ceno na zelo veliko številko).

// - cumulative: Matrika kumulativnih energij.
// - seams: 1D matrika velikosti (K * višina), kamor zaporedno shranimo X koordinate vseh K šivov.
// - curr_width: Trenutna efektivna širina slike.
// - curr_height: Višina slike.
// - orig_width: Originalna širina slike.
// - k: Število šivov, ki jih želimo poiskati v tem ciklu (običajno K=8).
void find_multiple_seams(float *cumulative, int *seams, int curr_width, int curr_height, int orig_width, int k) {
    for (int i = 0; i < k; i++) {
        float min_val = 1e9f;
        int min_c = 0;

        // Najdemo minimum v zgornji vrstici za trenutni šiv
        for (int c = 0; c < curr_width; c++) {
            if (cumulative[0 * orig_width + c] < min_val) {
                min_val = cumulative[0 * orig_width + c];
                min_c = c;
            }
        }
        
        seams[i * curr_height + 0] = min_c;
        cumulative[0 * orig_width + min_c] = 1e9f; // Označimo začetno točko kot uporabljeno

        // Sledimo poti navzdol
        for (int r = 0; r < curr_height - 1; r++) {
            int c = seams[i * curr_height + r];

            float m_left  = (c > 0)                  ? cumulative[(r + 1) * orig_width + c - 1] : 1e9f;
            float m_mid   =                            cumulative[(r + 1) * orig_width + c];
            float m_right = (c < curr_width - 1) ? cumulative[(r + 1) * orig_width + c + 1] : 1e9f;

            int next_c = c;
            float min_m = m_mid;

            if (m_left < min_m)  { min_m = m_left;  next_c = c - 1; }
            if (m_right < min_m) { min_m = m_right; next_c = c + 1; }

            seams[i * curr_height + r + 1] = next_c;
            
            // Da preprečimo prekrivanje oz. sekanje šivov, pot, ki smo jo pravkar izbrali 
            // in njene neposredne leve/desne sosede "zastrupimo" z ogromno vrednostjo.
            // Naslednji (i+1) šiv se bo tej poti avtomatsko izognil.
            cumulative[(r + 1) * orig_width + next_c] = 1e9f;
            if (next_c > 0) cumulative[(r + 1) * orig_width + next_c - 1] = 1e9f;
            if (next_c < curr_width - 1) cumulative[(r + 1) * orig_width + next_c + 1] = 1e9f;
        }
    }
}

//Odstranjevanje K šivov hkrati z uporabo logične maske
// Namesto da sliko premikamo s K zaporednimi `memmove` operacijami, ustvarimo masko tistih pikslov, ki jih brišemo.
// Nato vsako vrstico "stisnemo" v enem samem prehodu.

// - img: Kazalec na sliko.
// - seams: Tisti K šivi, ki jih želimo izbrisati.
// - mask: Začasna matrika (velikosti cele slike), ki drži 1, če piksel brišemo, in 0, če ga obdržimo.
// - curr_width: Trenutna efektivna širina slike pred brisanjem.
// - curr_height: Višina slike.
// - orig_width: Originalna širina slike za računanje preskokov v pomnilniku.
// - cpp: Število barvnih kanalov.
// - k: Število šivov, ki jih brišemo.
void remove_multiple_seams(unsigned char *img, int *seams, unsigned char *mask, int curr_width, int curr_height, int orig_width, int cpp, int k) {
    // 1. Resetiramo masko na 0. To lahko varno paraleliziramo.
    #pragma omp parallel for
    for (int i = 0; i < curr_height * orig_width; i++) {
        mask[i] = 0;
    }

    // 2. V masko zapišemo enice na tista mesta, kjer potekajo naši K šivi.
    #pragma omp parallel for
    for (int r = 0; r < curr_height; r++) {
        for (int i = 0; i < k; i++) {
            mask[r * orig_width + seams[i * curr_height + r]] = 1;
        }
    }

    // 3. Vzporedno stisnemo posamezne vrstice. Vsaka nit vzame svojo vrstico.
    #pragma omp parallel for
    for (int r = 0; r < curr_height; r++) {
        int dst_c = 0; // Kazalec destinacije (kamor bomo pisali ohranjene piksle)
        for (int c = 0; c < curr_width; c++) {
            // Če maska == 0, piksel OHRANIMO
            if (mask[r * orig_width + c] == 0) {
                // Če sta destinacija in trenutna lokacija različni, prekopiramo podatke v levo
                if (dst_c != c) {
                    for (int ch = 0; ch < cpp; ch++) {
                        img[(r * orig_width + dst_c) * cpp + ch] = img[(r * orig_width + c) * cpp + ch];
                    }
                }
                dst_c++; // Premaknemo destinacijo naprej samo ob veljavnem pikslu
            }
        }
    }
}

void run_benchmark(unsigned char *image_in, unsigned char *working_img, float *energy, 
                   float *cumulative, int *seams, unsigned char *mask, 
                   int width, int height, int cpp, int num_seams, size_t datasize, int K) {
    
    memcpy(working_img, image_in, datasize);
    int curr_width = width;
    
    // Začetek merjenja časa
    double start = omp_get_wtime();
    
    // Zanka sedaj preskakuje po K šivov naenkrat
    for (int s = 0; s < num_seams; s += K) {
        // Preprečimo brisanje preveč šivov v zadnjem koraku, če num_seams ni deljivo s K
        int current_k = (s + K <= num_seams) ? K : (num_seams - s);

        calculate_energy(working_img, energy, curr_width, height, width, cpp);
        calculate_cumulative_energy_triangles(energy, cumulative, curr_width, height, width, 32);
        
        find_multiple_seams(cumulative, seams, curr_width, height, width, current_k);
        remove_multiple_seams(working_img, seams, mask, curr_width, height, width, cpp, current_k);
        
        curr_width -= current_k; // Slika je za current_k pikslov ožja
    }

    // Konec merjenja časa
    double stop = omp_get_wtime();
    double elapsed = stop - start;
    
    printf("Time: %f s\n", elapsed);
}

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

    int K = 8; 
    int *seams = (int *)malloc(K * height * sizeof(int)); // Matrika za shranjevanje K poti
    unsigned char *mask = (unsigned char *)malloc(width * height * sizeof(unsigned char));

    if (!working_img || !energy || !cumulative || !seams || !mask) {
        printf("Error: Failed to allocate memory!\n");
        exit(EXIT_FAILURE);
    }

    // 4. Izvajanje algoritma z dodanimi parametri za paketno odstranjevanje
    run_benchmark(image_in, working_img, energy, cumulative, seams, mask, width, height, cpp, num_seams, datasize, K);
    
    // 5. Zlaganje slike skupaj in shranjevanje (odrežemo smeti z desne)
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