#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "utils.h"
#include "stb_image.h"

void calculate_energy(const unsigned char *img, float *energy, int curr_width, int curr_height, int orig_width, int cpp)
{
#pragma omp parallel for
    for (int r = 0; r < curr_height; r++)
    {
        for (int c = 0; c < curr_width; c++)
        {
            float total_energy = 0.0f;

            for (int ch = 0; ch < cpp; ch++)
            {
                int p_tl = get_pixel(img, r - 1, c - 1, curr_width, curr_height, orig_width, ch, cpp);
                int p_tc = get_pixel(img, r - 1, c, curr_width, curr_height, orig_width, ch, cpp);
                int p_tr = get_pixel(img, r - 1, c + 1, curr_width, curr_height, orig_width, ch, cpp);

                int p_ml = get_pixel(img, r, c - 1, curr_width, curr_height, orig_width, ch, cpp);
                int p_mr = get_pixel(img, r, c + 1, curr_width, curr_height, orig_width, ch, cpp);

                int p_bl = get_pixel(img, r + 1, c - 1, curr_width, curr_height, orig_width, ch, cpp);
                int p_bc = get_pixel(img, r + 1, c, curr_width, curr_height, orig_width, ch, cpp);
                int p_br = get_pixel(img, r + 1, c + 1, curr_width, curr_height, orig_width, ch, cpp);

                float gx = -p_tl - 2 * p_ml - p_bl + p_tr + 2 * p_mr + p_br;
                float gy = p_tl + 2 * p_tc + p_tr - p_bl - 2 * p_bc - p_br;

                total_energy += sqrtf(gx * gx + gy * gy);
            }
            energy[r * orig_width + c] = total_energy / (float)cpp;
        }
    }
}

// Paralelni izračun kumulativne energije s trikotniki
//
// Sliko razdelimo na horizontalne pasove višine B.
// Znotraj pasu se računanje razdeli na neodvisne "trikotnike", ki jih niti
// lahko obdelujejo vzporedno brez odvisnosti.

// - energy: Kazalec na matriko z izračunanimi energijami pikslov.
// - cumulative: Kazalec na matriko za seštevek najcenejše poti do dna.
// - curr_width: Trenutna efektivna širina slike.
// - curr_height: Višina slike v pikslih.
// - orig_width: Originalna širina slike v pomnilniku.
// - B: Višina posameznega horizontalnega pasu
void calculate_cumulative_energy_triangles(const float *energy, float *cumulative, int curr_width, int curr_height, int orig_width, int B)
{
// Inicializacija spodnje vrstice
#pragma omp parallel for
    for (int c = 0; c < curr_width; c++)
    {
        cumulative[(curr_height - 1) * orig_width + c] = energy[(curr_height - 1) * orig_width + c];
    }

    // Zunanja zanka preskakuje po pasovih višine B.

    // Te zanke ne moremo paralelizirati, saj višji pasovi strogo potrebujejo rezultate nižjih.
    for (int r_bottom = curr_height - 2; r_bottom >= 0; r_bottom -= B)
    {

        // Zgornja meja trenutnega pasu - ne smemo izven slike navzgor
        int r_top = r_bottom - B + 1;
        if (r_top < 0)
            r_top = 0;

        int current_B = r_bottom - r_top + 1; // Dejanska višina trenutnega pasu

        // Izračunamo število trikotnikov, ki prečkajo celotno širino slike.
        // Formula (W + 2B - 1) / (2B) + 1 zagotovi, da pokrijemo tudi desni rob.
        int num_triangles = (curr_width + 2 * current_B - 1) / (2 * current_B) + 1;

        // Izračun NAVZGOR obrnjenih trikotnikov
        // Vsak trikotnik ima vrh na spodnji vrstici pasu in se širi navzgor.

#pragma omp parallel for
        for (int t = 0; t < num_triangles; t++)
        {
            // Vsak trikotnik ima svoj "vrh" na specifični X koordinati
            int c_peak = t * 2 * current_B;

            for (int r = r_bottom; r >= r_top; r--)
            {
                // Širina trikotnika raste za 1 na vsaki strani, ko gremo višje.
                // spread = 0 v spodnji vrstici pasu (r == r_bottom) in raste navzgor.
                int spread = r_bottom - r;
                int c_start = c_peak - spread;
                int c_end = c_peak + spread;

                // Omejimo širjenje na dejanske meje slike (levi in desni rob)
                if (c_start < 0)
                    c_start = 0;
                if (c_end >= curr_width)
                    c_end = curr_width - 1;

                // Izračunamo poti samo znotraj meja trenutnega trikotnika
                for (int c = c_start; c <= c_end; c++)
                {
                    float m_left = (c > 0) ? cumulative[(r + 1) * orig_width + c - 1] : 1e9f;
                    float m_mid = cumulative[(r + 1) * orig_width + c];
                    float m_right = (c < curr_width - 1) ? cumulative[(r + 1) * orig_width + c + 1] : 1e9f;

                    float min_m = m_mid;
                    if (m_left < min_m)
                        min_m = m_left;
                    if (m_right < min_m)
                        min_m = m_right;

                    cumulative[r * orig_width + c] = energy[r * orig_width + c] + min_m;
                }
            }
        }

// Izračun NAVZDOL obrnjenih trikotnikov
// Sedaj imamo v pasu izračunane vrhove (A). Ostale so "vrzeli", ki po obliki
// spominjajo na navzdol obrnjene trikotnike. Njihova spodnja meja je široka,
// zgoraj pa se stikajo v točki. Ker se zanašajo na že izračunane robove A trikotnikov,
// jih zdaj lahko spet vzporedno zapolnimo.
#pragma omp parallel for
        for (int t = 0; t < num_triangles; t++)
        {
            for (int r = r_bottom; r >= r_top; r--)
            {
                // Vrzel leži točno MED dvema 'A' trikotnikoma.
                // Njihova širina se z višino manjša.
                int spread = r_bottom - r;
                int c_start = t * 2 * current_B + spread + 1;
                int c_end = (t + 1) * 2 * current_B - spread - 1;

                // Omejimo na meje slike
                if (c_start < 0)
                    c_start = 0;
                if (c_end >= curr_width)
                    c_end = curr_width - 1;

                // Računanje vrzeli. Če je c_start > c_end (npr. na vrhu trikotnika), se ta zanka preprosto preskoči.
                for (int c = c_start; c <= c_end; c++)
                {
                    float m_left = (c > 0) ? cumulative[(r + 1) * orig_width + c - 1] : 1e9f;
                    float m_mid = cumulative[(r + 1) * orig_width + c];
                    float m_right = (c < curr_width - 1) ? cumulative[(r + 1) * orig_width + c + 1] : 1e9f;

                    float min_m = m_mid;
                    if (m_left < min_m)
                        min_m = m_left;
                    if (m_right < min_m)
                        min_m = m_right;

                    cumulative[r * orig_width + c] = energy[r * orig_width + c] + min_m;
                }
            }
        }
    }
}

void find_seam(const float *cumulative, int *seam, int curr_width, int curr_height, int orig_width)
{
    float min_val = 1e9f;
    int min_c = 0;

    // Najdemo minimum v zgornji vrstici
    for (int c = 0; c < curr_width; c++)
    {
        if (cumulative[0 * orig_width + c] < min_val)
        {
            min_val = cumulative[0 * orig_width + c];
            min_c = c;
        }
    }
    seam[0] = min_c;

    // Sledimo poti navzdol
    for (int r = 0; r < curr_height - 1; r++)
    {
        int c = seam[r];

        float m_left = (c > 0) ? cumulative[(r + 1) * orig_width + c - 1] : 1e9f;
        float m_mid = cumulative[(r + 1) * orig_width + c];
        float m_right = (c < curr_width - 1) ? cumulative[(r + 1) * orig_width + c + 1] : 1e9f;

        int next_c = c;
        float min_m = m_mid;

        if (m_left < min_m)
        {
            min_m = m_left;
            next_c = c - 1;
        }
        if (m_right < min_m)
        {
            min_m = m_right;
            next_c = c + 1;
        }

        seam[r + 1] = next_c;
    }
}

void remove_seam(unsigned char *img, const int *seam, int curr_width, int curr_height, int orig_width, int cpp)
{
#pragma omp parallel for
    for (int r = 0; r < curr_height; r++)
    {
        int c_remove = seam[r];
        if (c_remove < curr_width - 1)
        {
            unsigned char *dst = &img[(r * orig_width + c_remove) * cpp];
            unsigned char *src = &img[(r * orig_width + c_remove + 1) * cpp];
            size_t bytes = (curr_width - 1 - c_remove) * cpp;
            memmove(dst, src, bytes);
        }
    }
}

// 5. MERJENJE ČASA
// Funkcija nadzoruje glavno zanko in meri zgolj čas računanja.

// - image_in: Originalna, nedotaknjena vhodna slika.
// - working_img: Začasni pomnilnik, kamor kopiramo sliko in jo nato manjšamo.
// - energy: Pred-alociran pomnilnik za matrice energij.
// - cumulative: Pred-alociran pomnilnik za kumulativne poti.
// - seam: Pred-alociran pomnilnik za pot trenutnega šiva.
// - width: Začetna širina slike.
// - height: Višina slike.
// - cpp: Število barvnih kanalov.
// - num_seams: Število šivov, ki jih mora algoritem poiskati in odstraniti.
// - datasize: Skupna velikost slike v bajtih (za hitro kopiranje z memcpy).
void run_benchmark(unsigned char *image_in, unsigned char *working_img, float *energy,
                   float *cumulative, int *seam, int width, int height, int cpp, int num_seams, size_t datasize)
{

    memcpy(working_img, image_in, datasize);
    int curr_width = width;

    double start = omp_get_wtime();

    for (int s = 0; s < num_seams; s++)
    {
        calculate_energy(working_img, energy, curr_width, height, width, cpp);

        calculate_cumulative_energy_triangles(energy, cumulative, curr_width, height, width, 32);
        find_seam(cumulative, seam, curr_width, height, width);
        remove_seam(working_img, seam, curr_width, height, width, cpp);

        curr_width--;
    }

    double stop = omp_get_wtime();
    double elapsed = stop - start;

    printf("Time: %f s\n", elapsed);
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        printf("Wrong number of arguments!\n");
        exit(EXIT_FAILURE);
    }

    char image_in_name[MAX_FILENAME];
    char image_out_name[MAX_FILENAME];
    snprintf(image_in_name, MAX_FILENAME, "%s", argv[1]);
    snprintf(image_out_name, MAX_FILENAME, "%s", argv[2]);
    int num_seams = atoi(argv[3]);

    int width, height, cpp;
    unsigned char *image_in = load_image_and_check(image_in_name, &width, &height, &cpp, num_seams);

    print_omp_info();

    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *working_img = (unsigned char *)malloc(datasize);
    float *energy = (float *)malloc(width * height * sizeof(float));
    float *cumulative = (float *)malloc(width * height * sizeof(float));
    int *seam = (int *)malloc(height * sizeof(int));

    if (!working_img || !energy || !cumulative || !seam)
    {
        printf("Error: Failed to allocate memory!\n");
        exit(EXIT_FAILURE);
    }

    run_benchmark(image_in, working_img, energy, cumulative, seam, width, height, cpp, num_seams, datasize);

    int final_width = width - num_seams;
    unsigned char *image_out = repack_image(working_img, width, final_width, height, cpp);

    if (image_out)
    {
        save_image(image_out_name, image_out, final_width, height, cpp);
    }

    stbi_image_free(image_in);
    free(working_img);
    free(image_out);
    free(energy);
    free(cumulative);
    free(seam);

    return 0;
}