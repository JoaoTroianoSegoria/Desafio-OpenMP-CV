/*---------------------------------------------------------------------------
 gerar_dados.cpp - cria o arquivo de entrada do exercício
-----------------------------------------------------------------------------
 O exercício diz: "cada linha contém a estatura e o peso de uma pessoa,
 separados por espaço". Como o arquivo não veio junto com o PDF, este
 programa cria um, com valores realistas:

   estatura: média 170 cm, desvio 10 cm      (curva normal)
   peso    : IMC médio 24, desvio 3,5  ->  peso = IMC x (estatura em m)²

 A SEMENTE é fixa (2026): rodar duas vezes gera EXATAMENTE o mesmo arquivo.
 Assim todo mundo do grupo trabalha com os mesmos dados e o mesmo resultado.

 Compilar: g++ -O2 gerar_dados.cpp -o gerar_dados
 Executar: ./gerar_dados 1000 pessoas.txt
-----------------------------------------------------------------------------*/
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <algorithm>

// Gerador de números pseudoaleatórios (o mesmo da aula 12_desempenho_mpi).
// Cada número depende do anterior -> com a mesma semente, a sequência se repete.
static uint64_t semente = 2026;
double aleatorio_0_1() {
    semente = semente * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(semente >> 11) / 9007199254740992.0;   // número em [0, 1)
}

// Número com distribuição normal (média 0, desvio 1) pelo método de Box-Muller.
double normal_padrao() {
    double u1 = aleatorio_0_1(), u2 = aleatorio_0_1();
    if (u1 < 1e-12) u1 = 1e-12;                          // evita log(0)
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

int main(int argc, char** argv) {
    int quantidade = (argc > 1) ? atoi(argv[1]) : 1000;
    const char* nome_arquivo = (argc > 2) ? argv[2] : "pessoas.txt";

    FILE* arquivo = std::fopen(nome_arquivo, "w");
    if (!arquivo) { std::perror(nome_arquivo); return 1; }

    for (int i = 0; i < quantidade; i++) {
        double estatura = std::clamp(170.0 + 10.0 * normal_padrao(), 140.0, 210.0);   // cm
        double imc      = std::clamp(24.0 + 3.5 * normal_padrao(), 16.0, 40.0);
        double peso     = imc * (estatura / 100.0) * (estatura / 100.0);              // kg
        std::fprintf(arquivo, "%.1f %.1f\n", estatura, peso);
    }
    std::fclose(arquivo);
    std::printf("Arquivo %s criado com %d pessoas (estatura peso).\n", nome_arquivo, quantidade);
    return 0;
}
