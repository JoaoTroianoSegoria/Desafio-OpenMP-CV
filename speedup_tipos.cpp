/*---------------------------------------------------------------------------
 speedup_tipos.cpp - Teste de speedup com os TIPOS de paralelização
-----------------------------------------------------------------------------
 Pega o laço central do exercício, a SOMA (Σ das estaturas, que é a base da
 média) e faz a mesma conta de vários jeitos, como nas aulas 006 e 007
 (o exemplo de Bhaskara com critical, atomic e reduction):

   sequencial          uma thread só (referência para o speedup)
   sem protecao        soma += x em paralelo SEM cuidado -> CONDIÇÃO DE CORRIDA
   critical            "banheiro com uma chave só": uma thread por vez na soma
   atomic              a soma vira uma operação indivisível do processador
   reduction static    cada thread soma na SUA cópia; junta no final (aula 007)
   reduction dynamic   igual, mas as iterações são entregues sob demanda
   reduction guided    igual, com pedaços que começam grandes e vão diminuindo

 Cada versão roda com 1, 2, 4, 8... threads e o programa imprime o speedup.

 Compilar: g++ -O2 -fopenmp speedup_tipos.cpp -o speedup_tipos
 Executar: ./speedup_tipos pessoas.txt            (usa 10 milhões de valores)
           ./speedup_tipos pessoas.txt 20000000   (escolhe a quantidade)
-----------------------------------------------------------------------------*/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>

const int REPETICOES = 3;   // cada tempo é a mediana de 3 execuções

/*------------------------- as sete versões da soma -------------------------*/
double soma_sequencial(const std::vector<double>& v) {
    double soma = 0.0;
    for (long i = 0; i < (long)v.size(); i++) soma += v[i];
    return soma;
}

double soma_sem_protecao(const std::vector<double>& v) {
    double soma = 0.0;
    #pragma omp parallel for
    for (long i = 0; i < (long)v.size(); i++)
        soma += v[i];               // ERRADO: ler-somar-gravar sem proteção (aula 006)
    return soma;
}

double soma_critical(const std::vector<double>& v) {
    double soma = 0.0;
    #pragma omp parallel for
    for (long i = 0; i < (long)v.size(); i++) {
        #pragma omp critical
        soma += v[i];               // certo, mas vira uma fila a CADA iteração
    }
    return soma;
}

double soma_atomic(const std::vector<double>& v) {
    double soma = 0.0;
    #pragma omp parallel for
    for (long i = 0; i < (long)v.size(); i++) {
        #pragma omp atomic
        soma += v[i];               // certo; fila mais rápida, mas ainda uma por iteração
    }
    return soma;
}

double soma_reduction_static(const std::vector<double>& v) {
    double soma = 0.0;
    #pragma omp parallel for reduction(+:soma) schedule(static)
    for (long i = 0; i < (long)v.size(); i++) soma += v[i];
    return soma;
}

double soma_reduction_dynamic(const std::vector<double>& v) {
    double soma = 0.0;
    #pragma omp parallel for reduction(+:soma) schedule(dynamic)
    for (long i = 0; i < (long)v.size(); i++) soma += v[i];
    return soma;
}

double soma_reduction_guided(const std::vector<double>& v) {
    double soma = 0.0;
    #pragma omp parallel for reduction(+:soma) schedule(guided)
    for (long i = 0; i < (long)v.size(); i++) soma += v[i];
    return soma;
}

/*--------------------------------------------------------------------------*/
struct Versao { const char* nome; double (*funcao)(const std::vector<double>&); };

// Roda 1 aquecimento + REPETICOES vezes e devolve a mediana do tempo.
double medir(double (*funcao)(const std::vector<double>&), const std::vector<double>& v, double& resultado) {
    resultado = funcao(v);
    std::vector<double> tempos;
    for (int rep = 0; rep < REPETICOES; rep++) {
        double t0 = omp_get_wtime();
        resultado = funcao(v);
        tempos.push_back(omp_get_wtime() - t0);
    }
    std::sort(tempos.begin(), tempos.end());
    return tempos[REPETICOES / 2];
}

int main(int argc, char** argv) {
    const char* nome_arquivo = (argc > 1) ? argv[1] : "pessoas.txt";
    long total = (argc > 2) ? atol(argv[2]) : 10000000L;

    // Lê só a estatura (primeira coluna) e repete até ter 'total' valores.
    FILE* arquivo = std::fopen(nome_arquivo, "r");
    if (!arquivo) { std::perror(nome_arquivo); return 1; }
    std::vector<double> lidos;
    double e, p;
    while (std::fscanf(arquivo, "%lf %lf", &e, &p) == 2) lidos.push_back(e);
    std::fclose(arquivo);
    if (lidos.empty()) { std::printf("Arquivo vazio.\n"); return 1; }

    std::vector<double> estatura(total);
    for (long i = 0; i < total; i++) estatura[i] = lidos[i % lidos.size()];

    std::vector<int> lista_threads;
    for (int t = 1; t <= omp_get_num_procs(); t *= 2) lista_threads.push_back(t);
    if (lista_threads.back() != omp_get_num_procs()) lista_threads.push_back(omp_get_num_procs());

    std::printf("Soma de %ld estaturas  |  nucleos logicos: %d  |  mediana de %d execucoes\n",
                total, omp_get_num_procs(), REPETICOES);

    double referencia;
    double t_seq = medir(soma_sequencial, estatura, referencia);
    std::printf("Sequencial: soma = %.1f   tempo = %.2f ms\n\n", referencia, t_seq * 1000);

    Versao versoes[] = {
        {"sem protecao",      soma_sem_protecao},
        {"critical",          soma_critical},
        {"atomic",            soma_atomic},
        {"reduction static",  soma_reduction_static},
        {"reduction dynamic", soma_reduction_dynamic},
        {"reduction guided",  soma_reduction_guided},
    };

    // Cabeçalho da tabela de speedup
    std::printf("SPEEDUP (tempo sequencial / tempo da versao)\n");
    std::printf("   %-18s", "versao \\ threads");
    for (int t : lista_threads) std::printf("%9d", t);
    std::printf("     resultado\n");

    for (const Versao& versao : versoes) {
        std::printf("   %-18s", versao.nome);
        bool sempre_certo = true;
        for (int t : lista_threads) {
            omp_set_num_threads(t);
            double resultado;
            double tempo = medir(versao.funcao, estatura, resultado);
            // Ordem diferente das somas muda só as últimas casas do double
            bool certo = std::fabs(resultado - referencia) <= 1e-9 * std::fabs(referencia);
            if (!certo) sempre_certo = false;
            std::printf("%8.2f%s", t_seq / tempo, certo ? " " : "*");
            std::fflush(stdout);
        }
        std::printf("     %s\n", sempre_certo ? "certo" : "ERRADO (*)");
    }
    std::printf("\n(*) = a soma deu diferente da sequencial (condicao de corrida)\n");
    std::printf("speedup > 1: mais rapido que o sequencial;  < 1: mais LENTO\n");
    return 0;
}
