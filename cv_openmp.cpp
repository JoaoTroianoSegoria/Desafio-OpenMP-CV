/*---------------------------------------------------------------------------
 cv_openmp.cpp - Exercício Final: Coeficiente de Variação com OpenMP
-----------------------------------------------------------------------------
 Lê a estatura (cm) e o peso (kg) de cada pessoa, agrupa em CLASSES
 (8 cm para estatura, 4 kg para peso) e calcula, para dados agrupados:

   Média ponderada      µ  = Σ (xi · fi) / N
   Desvio padrão pop.   σ  = √( Σ fi · (xi − µ)² / N )
   Coef. de variação    CV = (σ / µ) · 100%

   xi = ponto médio da classe i,  fi = frequência (quantas pessoas caem nela)

 O programa tem DUAS versões do mesmo cálculo (igual à aula 011, que tinha
 multiplicar_sequencial e multiplicar_paralelo):
   - calcular_sequencial(): uma thread faz tudo;
   - calcular_paralelo()  : os laços divididos entre as threads com OpenMP.
 Ele confere se as duas dão o MESMO resultado e mede o speedup.

 Compilar: g++ -O2 -fopenmp cv_openmp.cpp -o cv_openmp
 Executar: ./cv_openmp pessoas.txt           -> os 1000 dados do exercício
           ./cv_openmp pessoas.txt 10000     -> repete os dados 10000 vezes
                                                (10 milhões de pessoas)

 Por que repetir os dados? Com só 1000 pessoas o cálculo leva microssegundos:
 criar as threads custa MAIS do que fazer a conta, e o paralelo fica mais
 lento (aula 004: "com N = 10 o paralelo NÃO fica mais rápido").
 Repetir os dados multiplica todas as frequências pelo mesmo número, então
 a média, o desvio e o CV NÃO mudam; só o trabalho aumenta.
-----------------------------------------------------------------------------*/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>

const double CLASSE_ESTATURA = 8.0;   // cm
const double CLASSE_PESO     = 4.0;   // kg
const int    REPETICOES      = 11;    // cada tempo é a MEDIANA de 11 execuções

struct Resultado {
    double limite_inferior;    // onde começa a primeira classe
    double largura;            // tamanho da classe (8 cm ou 4 kg)
    std::vector<long> freq;    // freq[i] = quantas pessoas caem na classe i
    double media, desvio, cv;
};

// Ponto médio da classe i:  [lim + i·largura , lim + (i+1)·largura)  ->  centro
double ponto_medio(const Resultado& r, int i) {
    return r.limite_inferior + i * r.largura + r.largura / 2.0;
}

/*===========================================================================
 VERSÃO SEQUENCIAL
 ===========================================================================*/
Resultado calcular_sequencial(const std::vector<double>& valores, double largura) {
    Resultado r;
    long N = (long)valores.size();
    r.largura = largura;

    // 1) Menor e maior valor -> definem onde as classes começam e terminam
    double menor = valores[0], maior = valores[0];
    for (long i = 0; i < N; i++) {
        menor = std::min(menor, valores[i]);
        maior = std::max(maior, valores[i]);
    }
    r.limite_inferior = std::floor(menor);
    int num_classes = (int)((maior - r.limite_inferior) / largura) + 1;

    // 2) Frequência de cada classe (contar quantas pessoas caem em cada uma)
    std::vector<long> freq(num_classes, 0);
    for (long i = 0; i < N; i++) {
        int classe = (int)((valores[i] - r.limite_inferior) / largura);
        freq[classe]++;
    }
    r.freq = freq;

    // 3) Soma para a média ponderada: Σ xi·fi
    double soma = 0.0;
    for (int i = 0; i < num_classes; i++)
        soma += ponto_medio(r, i) * r.freq[i];
    r.media = soma / N;

    // 4) Soma para o desvio padrão: Σ fi·(xi − µ)²
    double soma_desvios = 0.0;
    for (int i = 0; i < num_classes; i++) {
        double d = ponto_medio(r, i) - r.media;
        soma_desvios += r.freq[i] * d * d;
    }
    r.desvio = std::sqrt(soma_desvios / N);
    r.cv = r.desvio / r.media * 100.0;
    return r;
}

/*===========================================================================
 VERSÃO PARALELA (OpenMP)
 ===========================================================================
 Variáveis, de forma geral:
   valores, N      -> COMPARTILHADAS e só LIDAS: seguro.
   i, classe, d    -> o índice do for e o que é declarado DENTRO do laço é PRIVADO.
   menor, maior, soma, soma_desvios, r.freq
                   -> COMPARTILHADAS e ESCRITAS por todas as threads: aqui mora a
                      CONDIÇÃO DE CORRIDA. Cada uma foi tratada de um jeito abaixo.
 ===========================================================================*/
Resultado calcular_paralelo(const std::vector<double>& valores, double largura) {
    Resultado r;
    long N = (long)valores.size();
    r.largura = largura;

    // 1) Menor e maior: reduction(min) e reduction(max) (aula 007).
    //    Cada thread acha o menor/maior do SEU pedaço numa cópia privada;
    //    no fim o OpenMP compara as cópias e fica com o menor/maior de todos.
    double menor = valores[0], maior = valores[0];
    #pragma omp parallel for reduction(min:menor) reduction(max:maior)
    for (long i = 0; i < N; i++) {
        menor = std::min(menor, valores[i]);
        maior = std::max(maior, valores[i]);
    }
    r.limite_inferior = std::floor(menor);
    int num_classes = (int)((maior - r.limite_inferior) / largura) + 1;

    // 2) Frequências. É o laço MAIS PESADO (passa por todas as N pessoas).
    //    PROBLEMA: r.freq[classe]++ em paralelo é corrida (duas threads podem
    //    contar na mesma classe ao mesmo tempo e uma contagem se perde).
    //    Opções da aula 006/007:
    //      - critical em cada ++  -> fila a cada pessoa (muito lento);
    //      - atomic em cada ++    -> fila menor, mas com só ~10 classes as
    //                                threads brigam o tempo todo pela mesma posição;
    //      - IDEIA DA REDUCTION (contar os votos por urna): cada thread conta
    //        num vetor PRÓPRIO (freq_local) e só no final soma no vetor geral.
    //        O critical passa a ser usado só ~10 vezes por thread, não N vezes.
    //    O arquivo speedup_tipos.cpp mede essas opções lado a lado.
    r.freq.assign(num_classes, 0);
    #pragma omp parallel
    {
        std::vector<long> freq_local(num_classes, 0);   // declarado aqui dentro: PRIVADO

        #pragma omp for schedule(static) nowait
        for (long i = 0; i < N; i++) {
            int classe = (int)((valores[i] - r.limite_inferior) / largura);
            freq_local[classe]++;                       // sem corrida: o vetor é só desta thread
        }
        // nowait: quem terminou já vai juntando, sem esperar as outras.

        #pragma omp critical
        {
            for (int c = 0; c < num_classes; c++)       // uma thread por vez soma no vetor geral
                r.freq[c] += freq_local[c];
        }
    }   // barreira implícita: daqui em diante r.freq está completo

    // 3) Soma da média ponderada: reduction(+) (aula 007).
    //    Obs.: este laço tem só ~10 iterações (uma por classe). O PDF pede para
    //    paralelizar a soma, mas aqui quase não há trabalho para dividir.
    double soma = 0.0;
    #pragma omp parallel for reduction(+:soma)
    for (int i = 0; i < num_classes; i++)
        soma += ponto_medio(r, i) * r.freq[i];
    r.media = soma / N;

    // 4) Soma do desvio padrão: reduction(+) de novo.
    //    Só pode começar depois da média pronta (depende de µ) -> são duas
    //    regiões paralelas separadas, uma depois da outra.
    double soma_desvios = 0.0;
    #pragma omp parallel for reduction(+:soma_desvios)
    for (int i = 0; i < num_classes; i++) {
        double d = ponto_medio(r, i) - r.media;
        soma_desvios += r.freq[i] * d * d;
    }
    r.desvio = std::sqrt(soma_desvios / N);
    r.cv = r.desvio / r.media * 100.0;
    return r;
}

/*===========================================================================
 Funções auxiliares
 ===========================================================================*/
// As duas versões deram o mesmo resultado? (frequências idênticas e CV igual;
// a soma em outra ordem pode mudar só a última casa do double: (a+b)+c != a+(b+c))
bool mesmo_resultado(const Resultado& a, const Resultado& b) {
    return a.freq == b.freq && std::fabs(a.cv - b.cv) <= 1e-9 * std::fabs(a.cv);
}

void imprimir_tabela(const char* nome, const char* unidade, const Resultado& r) {
    std::printf("\n%s (classes de %.0f %s)\n", nome, r.largura, unidade);
    std::printf("   classe             ponto medio   freq\n");
    for (size_t i = 0; i < r.freq.size(); i++) {
        double ini = r.limite_inferior + i * r.largura;
        std::printf("   [%6.1f , %6.1f)   %8.1f   %8ld\n", ini, ini + r.largura, ponto_medio(r, i), r.freq[i]);
    }
    std::printf("   media = %.4f %s   desvio padrao = %.4f %s   CV = %.4f %%\n",
                r.media, unidade, r.desvio, unidade, r.cv);
}

// Mede uma versão: 1 execução de aquecimento (descartada) + REPETICOES execuções.
// Devolve a MEDIANA dos tempos (menos sensível a uma execução "azarada").
double medir(Resultado (*calcular)(const std::vector<double>&, double),
             const std::vector<double>& estatura, const std::vector<double>& peso) {
    calcular(estatura, CLASSE_ESTATURA);                // aquecimento (cache, criação das threads)
    calcular(peso, CLASSE_PESO);
    std::vector<double> tempos;
    for (int rep = 0; rep < REPETICOES; rep++) {
        double t0 = omp_get_wtime();
        calcular(estatura, CLASSE_ESTATURA);
        calcular(peso, CLASSE_PESO);
        tempos.push_back(omp_get_wtime() - t0);
    }
    std::sort(tempos.begin(), tempos.end());
    return tempos[REPETICOES / 2];
}

int main(int argc, char** argv) {
    const char* nome_arquivo = (argc > 1) ? argv[1] : "pessoas.txt";
    int vezes = (argc > 2) ? atoi(argv[2]) : 1;
    if (vezes < 1) vezes = 1;

    // ---------- Leitura do arquivo (sequencial: fora da medição de tempo) ----------
    FILE* arquivo = std::fopen(nome_arquivo, "r");
    if (!arquivo) { std::perror(nome_arquivo); return 1; }
    std::vector<double> estatura, peso;
    double e, p;
    while (std::fscanf(arquivo, "%lf %lf", &e, &p) == 2) {
        estatura.push_back(e);
        peso.push_back(p);
    }
    std::fclose(arquivo);
    if (estatura.empty()) { std::printf("Arquivo vazio.\n"); return 1; }
    long pessoas_arquivo = (long)estatura.size();

    // Repete os dados (opcional) só para ter trabalho suficiente para medir.
    for (int k = 1; k < vezes; k++)
        for (long i = 0; i < pessoas_arquivo; i++) {
            estatura.push_back(estatura[i]);
            peso.push_back(peso[i]);
        }

    std::printf("Arquivo: %s  (%ld pessoas", nome_arquivo, pessoas_arquivo);
    if (vezes > 1) std::printf(" repetidas %dx = %zu pessoas", vezes, estatura.size());
    std::printf(")\nNucleos logicos: %d\n", omp_get_num_procs());

    // ---------- Resultados ----------
    Resultado est_seq  = calcular_sequencial(estatura, CLASSE_ESTATURA);
    Resultado peso_seq = calcular_sequencial(peso, CLASSE_PESO);
    imprimir_tabela("ESTATURA", "cm", est_seq);
    imprimir_tabela("PESO", "kg", peso_seq);

    std::printf("\n>>> CV da estatura = %.4f %%      CV do peso = %.4f %%\n", est_seq.cv, peso_seq.cv);

    // ---------- Desempenho: sequencial x paralelo com 1, 2, 4, 8... threads ----------
    std::vector<int> lista_threads;
    for (int t = 1; t <= omp_get_num_procs(); t *= 2) lista_threads.push_back(t);
    if (lista_threads.back() != omp_get_num_procs()) lista_threads.push_back(omp_get_num_procs());

    double t_seq = medir(calcular_sequencial, estatura, peso);
    std::printf("\nDESEMPENHO (mediana de %d execucoes, estatura + peso)\n", REPETICOES);
    std::printf("   versao       threads   tempo (ms)   speedup   eficiencia   resultado\n");
    std::printf("   sequencial      1     %10.4f      1.00      100.0%%      referencia\n", t_seq * 1000);

    for (int t : lista_threads) {
        omp_set_num_threads(t);
        double t_par = medir(calcular_paralelo, estatura, peso);
        Resultado est_par  = calcular_paralelo(estatura, CLASSE_ESTATURA);
        Resultado peso_par = calcular_paralelo(peso, CLASSE_PESO);
        bool ok = mesmo_resultado(est_seq, est_par) && mesmo_resultado(peso_seq, peso_par);
        double speedup = t_seq / t_par;
        std::printf("   paralelo     %4d     %10.4f    %6.2f      %5.1f%%      %s\n",
                    t, t_par * 1000, speedup, 100.0 * speedup / t, ok ? "igual" : "DIFERENTE!");
    }
    std::printf("\nspeedup = tempo sequencial / tempo paralelo     eficiencia = speedup / threads\n");
    return 0;
}
