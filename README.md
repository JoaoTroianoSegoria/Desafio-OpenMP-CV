# Desafio Final OpenMP — Coeficiente de Variação

Programação Paralela (CCO085) · IESB · Prof. Rodrigo Gonçalves

O programa calcula o **coeficiente de variação** da estatura e do peso de 1000 pessoas, com os dados agrupados em classes de **8 cm** e **4 kg**. Os laços de cálculo são paralelizados com OpenMP.

## Arquivos

| Arquivo | O que é |
|---|---|
| `cv_openmp.cpp` | O exercício: versão sequencial e versão paralela, conferência do resultado e tabela de speedup |
| `speedup_tipos.cpp` | Teste de speedup da soma feita sem proteção × critical × atomic × reduction (static, dynamic, guided) |
| `gerar_dados.cpp` | Cria `pessoas.txt` (1000 linhas `estatura peso`, sempre com a mesma semente) |
| `relatorio.pdf` | Relatório: o que foi paralelizado e por quê, resultados e análise de desempenho |
| `Makefile` | Compila e roda tudo |

## Como rodar (Linux do laboratório ou Docker)

```bash
make            # compila e cria pessoas.txt
make rodar      # resultado com as 1000 pessoas + tempo sequencial x paralelo
make grande     # mesmos dados repetidos 10000x (10 milhões) para o speedup aparecer
make tipos      # critical x atomic x reduction x schedules
```

Sem o Makefile:

```bash
g++ -O2 -fopenmp cv_openmp.cpp -o cv_openmp
./cv_openmp pessoas.txt
```

Para usar o Docker no Windows (PowerShell, dentro desta pasta):

```powershell
docker run --rm -it -v "${PWD}:/projeto" -w /projeto mandelbrot-pp bash
```

## Resultados (1000 pessoas)

| Variável | Média | Desvio padrão | CV |
|---|---|---|---|
| Estatura | 169,80 cm | 10,27 cm | **6,05 %** |
| Peso | 69,18 kg | 12,84 kg | **18,56 %** |

Com 1000 pessoas a versão paralela fica **mais lenta**, porque o cálculo leva microssegundos e criar as threads custa mais do que isso. Com 10 milhões de pessoas o speedup chega a cerca de **4× com 8 threads**. A explicação completa está no `relatorio.pdf`.
