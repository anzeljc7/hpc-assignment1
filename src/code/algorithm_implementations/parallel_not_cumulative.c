#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "utils.h"
#include "stb_image.h" 

// 1. KORAK: Paralelni izračun energije (Sobelov operator)
// Za vsak piksel izračuna njegovo "energijo" oziroma pomembnost.

// - img: Kazalec na vhodne piksle slike (1D zaporedje bajtov).
// - energy: Kazalec na 1D matriko (float), kamor zapišemo rezultat (energijo).
// - curr_width: Trenutna efektivna širina slike (ki se manjša z vsakim šivom).
// - curr_height: Višina slike v pikslih.
// - orig_width: Originalna širina slike v pomnilniku - potrebno za pravilen preskok v naslednjo vrstico.
// - cpp: Channels Per Pixel (število barvnih kanalov, npr. 3 za RGB).
void calculate_energy(const unsigned char *img, float *energy, int curr_width, int curr_height, int orig_width, int cpp) {
    // Zunanjo zanko po vrsticah lahko varno paraleliziramo, saj so izračuni pikslov neodvisni
    #pragma omp parallel for
    for (int r = 0; r < curr_height; r++) {
        for (int c = 0; c < curr_width; c++) {
            float total_energy = 0.0f;
            
            // Seštejemo energijo za vse barvne kanale (R, G, B)
            for (int ch = 0; ch < cpp; ch++) {
                // Branje 3x3 soseske piksla za Sobelov filter
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
            // Povprečimo energijo čez vse kanale in shranimo
            energy[r * orig_width + c] = total_energy / (float)cpp;
        }
    }
}

// 2. KORAK: Izračun kumulativne energije
// Funkcija potuje od spodnje vrstice proti zgornji. Za vsak piksel izračuna najcenejšo pot do dna slike.

// - energy: Kazalec na matriko z izračunanimi energijami pikslov (iz 1. koraka).
// - cumulative: Kazalec na matriko, kamor zapišemo seštevek najcenejše poti do dna.
// - curr_width: Trenutna efektivna širina slike.
// - curr_height: Višina slike v pikslih.
// - orig_width: Originalna širina slike za računanje indeksov v pomnilniku.
void calculate_cumulative_energy(const float *energy, float *cumulative, int curr_width, int curr_height, int orig_width) {
    // Spodnja vrstica: njena kumulativna energija je kar njena lastna energija
    #pragma omp parallel for
    for (int c = 0; c < curr_width; c++) {
        cumulative[(curr_height - 1) * orig_width + c] = energy[(curr_height - 1) * orig_width + c];
    }

    // Od predzadnje vrstice navzgor (zunanje zanke se ne paralelizira zaradi odvisnosti)
    for (int r = curr_height - 2; r >= 0; r--) {
        // Znotraj ene vrstice lahko računamo vzporedno
        for (int c = 0; c < curr_width; c++) {
            // Preverimo vrednosti treh spodnjih sosedov (levo, sredina, desno)
            // Če smo na robu slike, nastavimo ogromno vrednost (1e9f), da preprečimo izbiro
            float m_left  = (c > 0)                  ? cumulative[(r + 1) * orig_width + c - 1] : 1e9f;
            float m_mid   =                            cumulative[(r + 1) * orig_width + c];
            float m_right = (c < curr_width - 1) ? cumulative[(r + 1) * orig_width + c + 1] : 1e9f;

            // Najdemo minimum izmed teh treh sosedov
            float min_m = m_mid;
            if (m_left < min_m) min_m = m_left;
            if (m_right < min_m) min_m = m_right;

            // Prištejemo lastno energijo k najcenejši poti naprej
            cumulative[r * orig_width + c] = energy[r * orig_width + c] + min_m;
        }
    }
}

// 3. KORAK: Iskanje poti šiva od zgoraj navzdol
// Spustimo se od zgoraj navzdol in zapišemo X koordinate šiva.

// - cumulative: Matrika kumulativnih energij (iz 2. koraka).
// - seam: Matrika velikosti h (višina), kamor za vsako vrstico zapišemo X koordinato šiva.
// - curr_width: Trenutna efektivna širina slike.
// - curr_height: Višina slike.
// - orig_width: Originalna širina slike za računanje indeksov.
void find_seam(const float *cumulative, int *seam, int curr_width, int curr_height, int orig_width) {
    float min_val = 1e9f;
    int min_c = 0;

    // Najprej najdemo piksel z najmanjšo vrednostjo povsem zgoraj (vrstica 0)
    for (int c = 0; c < curr_width; c++) {
        if (cumulative[0 * orig_width + c] < min_val) {
            min_val = cumulative[0 * orig_width + c];
            min_c = c;
        }
    }
    seam[0] = min_c; // Shranimo začetno točko šiva

    // Sledimo poti navzdol - vedno izberemo tistega spodnjega soseda, ki ima najmanjšo vrednost
    for (int r = 0; r < curr_height - 1; r++) {
        int c = seam[r]; 

        float m_left  = (c > 0)                  ? cumulative[(r + 1) * orig_width + c - 1] : 1e9f;
        float m_mid   =                            cumulative[(r + 1) * orig_width + c];
        float m_right = (c < curr_width - 1) ? cumulative[(r + 1) * orig_width + c + 1] : 1e9f;

        int next_c = c;
        float min_m = m_mid;

        // Preverimo, ali je levi oz. desni sosed cenejši od sredinskega
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

// 4. KORAK: Pomik pikslov na levi po odstranitvi šiva
// Odstranimo šiv tako, da piksle desno od njega premaknemo za eno mesto v levo.

// - img: Kazalec na sliko, ki jo neposredno modificiramo.
// - seam: Matrika z X koordinatami šiva, ki ga brišemo (iz 3. koraka).
// - curr_width: Trenutna efektivna širina slike.
// - curr_height: Višina slike.
// - orig_width: Originalna širina slike za računanje preskokov v pomnilniku.
// - cpp: Število barvnih kanalov (koliko bajtov zavzame en piksel).
void remove_seam(unsigned char *img, const int *seam, int curr_width, int curr_height, int orig_width, int cpp) {
    // Vsaka vrstica briše svoj del šiva, zato lahko to naredimo vzporedno
    #pragma omp parallel for
    for (int r = 0; r < curr_height; r++) {
        int c_remove = seam[r]; // Kateri piksel v tej vrstici moramo odstraniti
        
        // Če piksel ni čisto na desnem robu, moramo premikati preostale piksle
        if (c_remove < curr_width - 1) {
            unsigned char *dst = &img[(r * orig_width + c_remove) * cpp];
            unsigned char *src = &img[(r * orig_width + c_remove + 1) * cpp];
            size_t bytes = (curr_width - 1 - c_remove) * cpp; // Število bajtov do konca vrstice
            
            // spominska bloka se prekrivata
            memmove(dst, src, bytes);
        }
    }
}

// 5. MERJENJE ČASA
// Funkcija nadzoruje glavno zanko in meri zgolj čas računanja.

// - image_in: Originalna, nedotaknjena vhodna slika.
// - working_img: Začasni pomnilnik, kamor kopiramo sliko in jo nato manjšamo.
// - energy: Pred-alociran pomnilnik za energijo.
// - cumulative: Pred-alociran pomnilnik za kumulativno energijo.
// - seam: Pred-alociran pomnilnik za pot šiva.
// - width: Začetna širina slike.
// - height: Višina slike.
// - cpp: Število barvnih kanalov.
// - num_seams: Število šivov, ki jih mora algoritem poiskati in odstraniti.
// - datasize: Skupna velikost slike v bajtih.

void run_benchmark(unsigned char *image_in, unsigned char *working_img, float *energy, 
                   float *cumulative, int *seam, int width, int height, int cpp, int num_seams, size_t datasize) {
    
    // Na začetku si ustvarimo delovno kopijo slike
    memcpy(working_img, image_in, datasize);
    int curr_width = width;
    
    // Beleženje začetnega časa
    double start = omp_get_wtime();
    
    for (int s = 0; s < num_seams; s++) {
        calculate_energy(working_img, energy, curr_width, height, width, cpp);
        calculate_cumulative_energy(energy, cumulative, curr_width, height, width);
        find_seam(cumulative, seam, curr_width, height, width);
        remove_seam(working_img, seam, curr_width, height, width, cpp);
        
        curr_width--; // Po odstranitvi šiva je slika efektivno ožja za 1 piksel
    }

    double stop = omp_get_wtime();
    double elapsed = stop - start;
    
    printf("Time: %f s\n", elapsed);
}


// - argc: Število argumentov iz komandne vrstice.
// - argv: Matrika stringov (argumenti komandne vrstice).
int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Wrong number of arguments!\n");
        exit(EXIT_FAILURE);
    }

    // Branje poti in števila šivov iz argumentov
    char image_in_name[MAX_FILENAME];
    char image_out_name[MAX_FILENAME];
    snprintf(image_in_name, MAX_FILENAME, "%s", argv[1]);
    snprintf(image_out_name, MAX_FILENAME, "%s", argv[2]);
    int num_seams = atoi(argv[3]);

    int width, height, cpp;
    unsigned char *image_in = load_image_and_check(image_in_name, &width, &height, &cpp, num_seams);
    
    print_omp_info();

    // Alokacija spomina za matrike
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *working_img = (unsigned char *)malloc(datasize);
    float *energy = (float *)malloc(width * height * sizeof(float));
    float *cumulative = (float *)malloc(width * height * sizeof(float));
    int *seam = (int *)malloc(height * sizeof(int));

    // Preverjanje uspele alokacije
    if (!working_img || !energy || !cumulative || !seam) {
        printf("Error: Failed to allocate memory!\n");
        exit(EXIT_FAILURE);
    }

    // Glavni izračun
    run_benchmark(image_in, working_img, energy, cumulative, seam, width, height, cpp, num_seams, datasize);

    // Na koncu zložimo piksle skupaj, da se znebimo "praznega" prostora na desni
    int final_width = width - num_seams;
    unsigned char *image_out = repack_image(working_img, width, final_width, height, cpp);
    
    // Shranimo izhodno sliko
    if (image_out) {
        save_image(image_out_name, image_out, final_width, height, cpp);
    }

    // Sproščanje spomina
    stbi_image_free(image_in);
    free(working_img);
    free(image_out);
    free(energy);
    free(cumulative);
    free(seam);

    return 0;
}